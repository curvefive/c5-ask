"""Local daemon that keeps the `codex` and `claude` CLIs warm, plus its client and controls.

Protocol over a Unix socket (one request per connection, JSON lines):
  -> {"op": "ask", "provider": "codex" | "claude-code", "model": str, "messages": [...]}
  <- {"delta": str} ... then {"done": true} or {"error": str}
  -> {"op": "status"} / {"op": "stop"}   <- one JSON object
Clients must keep their write side open until the answer ends: EOF cancels the request.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import plistlib
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from c5_ask import __version__
from c5_ask.backends import ClaudeCodeBackend, CodexBackend
from c5_ask.providers import Message, ProviderError, validate_messages, validate_model

log = logging.getLogger("c5_ask.daemon")

START_TIMEOUT = 20  # seconds to wait for an auto-started daemon to accept connections
CLIENT_TIMEOUT = 150
LAUNCHD_LABEL = "com.curvefive.c5-ask"
DAEMON_PROVIDERS = ("codex", "claude-code")


MAX_LOG_BYTES = 5 * 1024 * 1024


def private_dir(path: Path) -> Path:
    """Create `path` (mode 0700) and refuse to use it unless only this user can access it.

    The socket, log and history live here; anyone who could write to the directory could
    swap the socket and receive every question (including piped files).
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise ProviderError(f"refusing to use {path}: not a directory owned by you")
    if st.st_mode & 0o077:
        path.chmod(0o700)
    return path


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return private_dir(Path(base) / "c5-ask")


def socket_path() -> Path:
    path = state_dir() / "daemon.sock"
    if len(os.fsencode(path)) < 100:  # sun_path is 104 bytes on macOS, 108 on Linux
        return path
    # Deep state dir: use a private dir under the temp dir instead. Hash the state dir so
    # separate state dirs never share a daemon.
    tag = hashlib.sha256(os.fsencode(path)).hexdigest()[:10]
    short = private_dir(Path(tempfile.gettempdir()) / f"c5-ask-{os.getuid()}")
    return short / f"{tag}.sock"


def log_path() -> Path:
    return state_dir() / "daemon.log"


def default_models() -> dict[str, str]:
    from c5_ask.providers import PROVIDERS

    return {name: PROVIDERS[name].default_model for name in DAEMON_PROVIDERS}


# --------------------------------------------------------------------------- server


class Daemon:
    def __init__(self) -> None:
        workdir = state_dir() / "work"  # empty dir: no project instructions get picked up
        workdir.mkdir(exist_ok=True)
        self.codex = CodexBackend(workdir)
        self.claude = ClaudeCodeBackend(workdir)
        self.started = time.time()
        self.stopping = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        async def send(obj: dict) -> None:
            writer.write(json.dumps(obj).encode() + b"\n")
            await writer.drain()

        try:
            request = json.loads(await reader.readline() or b"{}")
            op = request.get("op") if isinstance(request, dict) else None
            if op == "status":
                await send(self.status())
            elif op == "stop":
                await send({"ok": True})
                self.stopping.set()
            elif op == "ask":
                await self.ask_until_disconnect(request, send, reader)
            else:
                await send({"error": f"unknown op: {op!r}"})
        except (ConnectionError, BrokenPipeError):
            log.info("client disconnected")  # ponytail: codex turn keeps running until done
        except Exception as exc:  # never let one request take the daemon down
            log.exception("request failed")
            with contextlib.suppress(Exception):
                await send({"error": str(exc)})
        finally:
            writer.close()

    async def ask_until_disconnect(self, request: dict, send, reader) -> None:
        """Answer, but cancel as soon as the client hangs up (Ctrl-C, `| head`), even while
        the model is still thinking and nothing is being written to the socket."""
        answer = asyncio.create_task(self.ask(request, send))
        hangup = asyncio.create_task(reader.read())  # the client sends nothing after the request
        await asyncio.wait({answer, hangup}, return_when=asyncio.FIRST_COMPLETED)
        if not answer.done():
            log.info("client disconnected; cancelling request")
            answer.cancel()
        hangup.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await answer

    async def ask(self, request: dict, send) -> None:
        provider, model = request.get("provider"), request.get("model")
        messages = request.get("messages")
        try:
            if provider not in DAEMON_PROVIDERS:
                raise ProviderError(f"unknown provider: {provider!r}")
            validate_model(model)
            validate_messages(messages)
        except ProviderError as exc:
            await send({"error": f"bad request: {exc}"})
            return
        backend = self.codex if provider == "codex" else self.claude
        started = time.monotonic()
        try:
            # aclosing: if the client disconnects, the backend cleans up right away
            # (kills claude / interrupts the codex turn) instead of whenever GC runs.
            async with contextlib.aclosing(backend.ask(model, messages)) as chunks:
                async for chunk in chunks:
                    await send({"delta": chunk})
            await send({"done": True})
            log.info("%s/%s answered in %.2fs", provider, model, time.monotonic() - started)
        except ProviderError as exc:
            log.warning("%s/%s failed: %s", provider, model, exc)
            await send({"error": str(exc)})

    def status(self) -> dict:
        return {
            "pid": os.getpid(),
            "version": __version__,
            "uptime_s": round(time.time() - self.started),
            "codex": self.codex.status(),
            "claude-code": self.claude.status(),
        }

    async def warm_up(self) -> None:
        for name, coro in (
            ("codex", self.codex.start()),
            ("claude-code", self.claude.prewarm(default_models()["claude-code"])),
        ):
            try:
                await coro
            except Exception as exc:  # a missing CLI must not stop the other backend
                log.warning("could not warm %s: %s", name, exc)

    async def serve(self) -> None:
        path = socket_path()
        path.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(self.handle, path=str(path), limit=2**24)
        path.chmod(0o600)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stopping.set)
        log.info("listening on %s (pid %s)", path, os.getpid())
        asyncio.create_task(self.warm_up())
        async with server:
            await self.stopping.wait()
        await self.codex.stop()
        await self.claude.stop()
        path.unlink(missing_ok=True)
        log.info("stopped")


def run_daemon() -> int:
    """Run in the foreground (what launchd and auto-start execute)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lock = open(lock_path(), "a")  # noqa: SIM115 - held for process lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("daemon already running")
        return 0
    _rotate_log()
    asyncio.run(Daemon().serve())
    return 0


def _rotate_log() -> None:
    """Keep the log bounded: our stdout/stderr point at it, so truncate in place."""
    with contextlib.suppress(OSError):
        if log_path().stat().st_size > MAX_LOG_BYTES:
            os.truncate(log_path(), 0)


# --------------------------------------------------------------------------- client


def _connect(timeout: float = CLIENT_TIMEOUT) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path()))
    except OSError:
        sock.close()
        raise
    return sock


def _request(obj: dict, timeout: float = 5) -> dict:
    with _connect(timeout) as sock, sock.makefile("rwb") as f:
        f.write(json.dumps(obj).encode() + b"\n")
        f.flush()
        return json.loads(f.readline() or b"{}")


def daemon_status() -> dict | None:
    """The running daemon's status, or None if none is reachable."""
    try:
        return _request({"op": "status"}, timeout=2)
    except (OSError, ValueError):
        return None


def is_running() -> bool:
    return daemon_status() is not None


def lock_path() -> Path:
    return state_dir() / "daemon.lock"


def _daemon_process_alive() -> bool:
    """True while a daemon holds the lock: serving, or still shutting down its CLIs.

    The socket goes away first on shutdown, so "not reachable" alone doesn't mean a new
    daemon can start yet.
    """
    with open(lock_path(), "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _wait_stopped(timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while (is_running() or _daemon_process_alive()) and time.monotonic() < deadline:
        time.sleep(0.1)


def spawn_background() -> subprocess.Popen:
    with open(log_path(), "ab") as log_file:  # the child keeps its own copy of the fd
        return subprocess.Popen(
            [sys.executable, "-m", "c5_ask", "--daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )


def ensure_running() -> None:
    status = daemon_status()
    if status and status.get("version") == __version__:
        return
    if status:
        # Left over from before an upgrade: replace it so it runs the installed code.
        log.info("restarting q daemon %s -> %s", status.get("version"), __version__)
        with contextlib.suppress(OSError, ValueError):
            _request({"op": "stop"})
        _wait_stopped()
    if sys.stderr.isatty():
        print("q: starting background service (first run only)...", file=sys.stderr)
    proc = spawn_background()
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if is_running():
            return
        code = proc.poll()
        if code == 0 and not _daemon_process_alive():
            # It found the lock taken, but whoever held it has exited since (a daemon that
            # was still shutting down): start again. If another q's daemon is starting, it
            # holds the lock and we just keep waiting for it to listen.
            proc = spawn_background()
        elif code not in (None, 0):
            break
        time.sleep(0.1)
    lines = log_path().read_text(errors="replace").strip().splitlines()
    raise ProviderError(
        f"q daemon did not start: {lines[-1] if lines else 'no log'} (log: {log_path()})"
    )


def stream_via_daemon(provider: str, model: str, messages: list[Message]) -> Iterator[str]:
    ensure_running()
    request = {"op": "ask", "provider": provider, "model": model, "messages": messages}
    try:
        with _connect() as sock, sock.makefile("rwb") as f:
            f.write(json.dumps(request).encode() + b"\n")
            f.flush()
            for line in f:
                msg = json.loads(line)
                if "delta" in msg:
                    yield msg["delta"]
                elif msg.get("done"):
                    return
                elif "error" in msg:
                    raise ProviderError(msg["error"])
    except TimeoutError:
        raise ProviderError(f"no reply from q daemon in {CLIENT_TIMEOUT}s") from None
    except (OSError, ValueError) as exc:
        raise ProviderError(f"lost connection to q daemon ({exc}); see {log_path()}") from None
    raise ProviderError(f"q daemon closed the connection; see {log_path()}")


# --------------------------------------------------------------------------- controls


def launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def control(action: str) -> int:
    """Handle `q --daemon ACTION`; returns an exit code."""
    if action == "run":
        return run_daemon()
    if action == "start":
        ensure_running()
        print("q daemon running")
        return 0
    if action == "stop":
        if not is_running():
            print("q daemon not running")
            return 0
        _request({"op": "stop"})
        _wait_stopped()
        print("q daemon stopped")
        return 0
    if action == "restart":
        control("stop")
        return control("start")
    if action == "status":
        if not is_running():
            print("q daemon not running")
            return 1
        print(json.dumps(_request({"op": "status"}), indent=2))
        return 0
    if action == "install":
        return _launchd_install()
    if action == "uninstall":
        return _launchd_uninstall()
    raise ValueError(action)


def _launchd_install() -> int:
    if sys.platform != "darwin":
        print(
            "q: --daemon install is macOS only; the daemon auto-starts on first use anyway",
            file=sys.stderr,
        )
        return 1
    _launchd_uninstall(quiet=True)
    control("stop")
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [sys.executable, "-m", "c5_ask", "--daemon", "run"],
        # launchd starts with a bare PATH; keep the one that can find codex/claude/node.
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},  # restart after crashes, not after `stop`
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
    }
    path = launchd_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(plist))
    subprocess.run(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    print(f"installed {path}; the daemon now starts at login")
    return 0


def _launchd_uninstall(quiet: bool = False) -> int:
    path = launchd_plist_path()
    if path.exists():
        subprocess.run(
            ["/bin/launchctl", "bootout", f"gui/{os.getuid()}", str(path)], capture_output=True
        )
        path.unlink()
        if not quiet:
            print(f"removed {path}")
    elif not quiet:
        print("q daemon is not installed")
    return 0
