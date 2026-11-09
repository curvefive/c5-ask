"""Subscription backends: drive the logged-in `codex` and `claude` CLIs as warm subprocesses.

These run inside the daemon (see daemon.py) so the CLIs' multi-second startup is paid ahead
of time instead of on every question.

Security model: questions may contain untrusted text (piped logs, files), so both CLIs run
with every tool that could touch the filesystem, the network or other programs disabled.
A prompt injection can then only influence the answer text, never take actions.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

from c5_ask.providers import SYSTEM_PROMPT, Message, ProviderError

log = logging.getLogger("c5_ask.daemon")

STREAM_LIMIT = 16 * 1024 * 1024  # max bytes per JSON line from a CLI
IDLE_TIMEOUT = 120  # seconds without output before a request fails
MAX_SPARE_MODELS = 2  # warm claude processes are kept for at most this many models

# codex features that give the model tools. Disabled when the installed codex knows them
# (unknown names make codex refuse to start, and names change between versions).
CODEX_TOOL_FEATURES = (
    "hooks", "shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins",
    "browser_use", "browser_use_external", "in_app_browser", "computer_use",
    "image_generation", "view_image", "skill_search", "skill_mcp_dependency_install",
    "tool_suggest", "sleep_tool", "multi_agent", "multi_agent_v2", "goals",
    "in_app_local_automation", "code_mode",
)  # fmt: skip


def to_prompt(messages: list[Message]) -> str:
    """Flatten a conversation into one prompt; each request starts a fresh CLI session."""
    *previous, last = messages
    if not previous:
        return last["content"]
    turns = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in previous)
    return (
        f"<previous_conversation>\n{turns}\n</previous_conversation>\n\n"
        f"Follow-up question:\n{last['content']}"
    )


def find_cli(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ProviderError(f"`{name}` is not installed or not on PATH")
    return path


# --------------------------------------------------------------------------- codex


def codex_known_features(codex: str) -> set[str]:
    """Feature flags this codex version accepts for --disable (empty if it can't tell)."""
    try:
        out = subprocess.run(
            [codex, "features", "list"], capture_output=True, text=True, timeout=20, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not list codex features: %s", exc)
        return set()
    return {
        parts[0] for line in out.splitlines() if len(parts := line.split()) >= 2
        if parts[1] != "removed"
    }  # fmt: skip


class CodexBackend:
    """A long-lived `codex app-server` (JSON-RPC on stdio); one ephemeral thread per question."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.proc: asyncio.subprocess.Process | None = None
        self.ids = itertools.count(1)
        self.pending: dict[int, asyncio.Future] = {}
        self.threads: dict[str, asyncio.Queue] = {}
        self.lock = asyncio.Lock()

    def command(self, known_features: set[str]) -> list[str]:
        disable = [a for f in CODEX_TOOL_FEATURES if f in known_features for a in ("--disable", f)]
        return [
            find_cli("codex"), "app-server", *disable,
            "-c", 'web_search="disabled"',
            "-c", "mcp_servers={}",
            # The user's custom provider (e.g. a proxy) may inject its own instructions.
            "-c", 'model_provider="openai"',
        ]  # fmt: skip

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        async with self.lock:
            if self.running:
                return
            known = await asyncio.to_thread(codex_known_features, find_cli("codex"))
            if "shell_tool" not in known:
                # Can't confirm the shell is off: refuse rather than run with tools enabled.
                raise ProviderError("could not verify codex tools can be disabled; see daemon log")
            proc = await asyncio.create_subprocess_exec(
                *self.command(known),
                cwd=self.workdir,
                limit=STREAM_LIMIT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,  # the daemon log
            )
            self.proc = proc
            asyncio.create_task(self._read_loop(proc))
            try:
                await self._call(
                    "initialize",
                    {
                        "clientInfo": {"name": "c5-ask", "title": "c5-ask", "version": "1"},
                        "capabilities": None,
                    },
                )
                await self._send({"method": "initialized"})
            except BaseException:
                await self._kill()
                raise
            log.info("codex app-server started (pid %s)", proc.pid)

    async def _kill(self) -> None:
        if self.proc and self.proc.returncode is None:
            _kill(self.proc)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.proc.wait(), 5)
        self.proc = None

    async def stop(self) -> None:
        if self.running:
            self.proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.proc.wait(), 5)
        await self._kill()

    async def _send(self, message: dict) -> None:
        if not self.running or not self.proc.stdin:
            raise ProviderError("codex app-server is not running")
        try:
            self.proc.stdin.write(json.dumps(message).encode() + b"\n")
            await self.proc.stdin.drain()
        except (ConnectionError, BrokenPipeError) as exc:
            raise ProviderError(f"codex app-server exited: {exc}") from None

    async def _call(self, method: str, params: dict) -> dict:
        request_id = next(self.ids)
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, IDLE_TIMEOUT)
        except TimeoutError:
            raise ProviderError(f"codex did not answer {method} in {IDLE_TIMEOUT}s") from None
        finally:
            self.pending.pop(request_id, None)

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout
        try:
            async for line in proc.stdout:
                with contextlib.suppress(ValueError):
                    await self._dispatch(json.loads(line))
        except Exception:
            log.exception("codex output could not be read")
            _kill(proc)
        finally:
            # Process gone: fail everything waiting on it; the next request restarts it.
            log.warning("codex app-server exited (code %s)", proc.returncode)
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ProviderError("codex app-server exited"))
            for queue in self.threads.values():
                error = {"error": {"message": "codex app-server exited"}, "willRetry": False}
                queue.put_nowait({"method": "error", "params": error})

    async def _dispatch(self, msg: object) -> None:
        if not isinstance(msg, dict):
            return
        if "id" in msg and "method" in msg:
            # Server -> client request (approvals, user input). q never grants anything.
            with contextlib.suppress(ProviderError):
                await self._send(
                    {"id": msg["id"], "error": {"code": -32601, "message": "not supported"}}
                )
        elif "id" in msg:
            future = self.pending.get(msg["id"])
            if future and not future.done():
                if "error" in msg:
                    future.set_exception(ProviderError(_rpc_error(msg["error"])))
                else:
                    future.set_result(msg.get("result") or {})
        else:
            thread_id = (msg.get("params") or {}).get("threadId")
            if thread_id in self.threads:
                self.threads[thread_id].put_nowait(msg)

    async def ask(self, model: str, messages: list[Message]) -> AsyncIterator[str]:
        await self.start()
        result = await self._call(
            "thread/start",
            {
                "model": model,
                "cwd": str(self.workdir),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": True,
                "baseInstructions": SYSTEM_PROMPT,
            },
        )
        thread_id = result["thread"]["id"]
        queue: asyncio.Queue = asyncio.Queue()
        self.threads[thread_id] = queue
        turn_id = None
        finished = False
        try:
            turn = await self._call(
                "turn/start",
                {
                    "threadId": thread_id,
                    "effort": "low",
                    "input": [{"type": "text", "text": to_prompt(messages), "text_elements": []}],
                },
            )
            turn_id = (turn.get("turn") or {}).get("id")
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), IDLE_TIMEOUT)
                except TimeoutError:
                    raise ProviderError(f"codex sent nothing for {IDLE_TIMEOUT}s") from None
                method, params = msg.get("method"), msg.get("params") or {}
                if method == "item/agentMessage/delta":
                    yield params.get("delta", "")
                elif method == "error" and not params.get("willRetry"):
                    finished = True
                    raise ProviderError(_rpc_error(params.get("error") or {}))
                elif method == "turn/completed":
                    finished = True
                    turn = params.get("turn") or {}
                    if turn.get("status") == "failed":
                        raise ProviderError(_rpc_error(turn.get("error") or {}))
                    return
        finally:
            self.threads.pop(thread_id, None)
            if not finished and turn_id and self.running:
                # Client went away mid-answer (Ctrl-C): stop spending the user's quota.
                interrupt = {"threadId": thread_id, "turnId": turn_id}
                with contextlib.suppress(Exception):
                    await self._send(
                        {"id": next(self.ids), "method": "turn/interrupt", "params": interrupt}
                    )

    def status(self) -> dict:
        return {"running": self.running, "pid": self.proc.pid if self.running else None}


def _rpc_error(error: dict) -> str:
    message = error.get("message") or "unknown error"
    with contextlib.suppress(ValueError, TypeError, AttributeError):
        # Upstream errors arrive as JSON text: {"detail": ..., "error": {"message": ...}}
        detail = json.loads(message)
        message = detail.get("error", {}).get("message") or detail.get("detail") or message
    return f"codex: {message}"


# --------------------------------------------------------------------------- claude


class ClaudeCodeBackend:
    """Keeps a pre-started `claude -p` process per model; each answers exactly one question.

    A stream-json `claude` session keeps its own context, so reusing one would leak earlier
    questions into later ones. Instead each question takes the spare and a replacement starts
    in the background, which hides the startup time without sharing state.
    """

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.spares: dict[str, asyncio.subprocess.Process] = {}  # insertion order = LRU
        self.locks: dict[str, asyncio.Lock] = {}
        # Models that answered at least once. Only these get a spare ahead of time, so a
        # mistyped `-m` doesn't leave an idle process behind.
        self.working_models: set[str] = set()

    def command(self, model: str) -> list[str]:
        return [
            find_cli("claude"), "-p", "--model", model,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--verbose", "--include-partial-messages",
            # No tools, MCP servers, settings, hooks, skills or saved sessions: just answers.
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--disable-slash-commands", "--no-session-persistence",
            "--system-prompt", SYSTEM_PROMPT,
        ]  # fmt: skip

    async def _spawn(self, model: str) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            *self.command(model),
            cwd=self.workdir,
            limit=STREAM_LIMIT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # the daemon log; an unread pipe could fill up and block claude
        )
        log.info("claude spare started for %s (pid %s)", model, proc.pid)
        return proc

    def _lock(self, model: str) -> asyncio.Lock:
        return self.locks.setdefault(model, asyncio.Lock())

    async def _ensure_spare(self, model: str) -> None:
        proc = self.spares.pop(model, None)  # re-insert last: most recently used
        if proc is None or proc.returncode is not None:
            proc = await self._spawn(model)
        self.spares[model] = proc
        while len(self.spares) > MAX_SPARE_MODELS:
            _kill(self.spares.pop(next(iter(self.spares))))

    async def prewarm(self, model: str) -> None:
        self.working_models.add(model)  # explicitly requested (the daemon's default model)
        async with self._lock(model):
            await self._ensure_spare(model)

    async def _take(self, model: str) -> asyncio.subprocess.Process:
        # Under the lock so concurrent questions never get (or spawn) the same spare twice.
        async with self._lock(model):
            await self._ensure_spare(model)
            proc = self.spares.pop(model)
        if model in self.working_models:
            asyncio.create_task(self._replace(model))
        return proc

    async def _replace(self, model: str) -> None:
        try:
            async with self._lock(model):
                await self._ensure_spare(model)
        except Exception:
            log.exception("could not start replacement claude for %s", model)

    async def stop(self) -> None:
        for proc in self.spares.values():
            _kill(proc)
        self.spares.clear()

    async def ask(self, model: str, messages: list[Message]) -> AsyncIterator[str]:
        proc = await self._take(model)
        assert proc.stdin and proc.stdout
        streamed = False
        try:
            user = {"type": "user", "message": {"role": "user", "content": to_prompt(messages)}}
            try:
                proc.stdin.write(json.dumps(user).encode() + b"\n")
                await proc.stdin.drain()
            except (ConnectionError, BrokenPipeError):
                raise ProviderError("claude exited before the question was sent") from None
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), IDLE_TIMEOUT)
                except TimeoutError:
                    raise ProviderError(f"claude sent nothing for {IDLE_TIMEOUT}s") from None
                if not line:
                    raise ProviderError("claude exited without answering; see daemon log")
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                text = claude_event_text(event)
                if text:
                    streamed = True
                    yield text
                elif isinstance(event, dict) and event.get("type") == "result":
                    if event.get("is_error") or event.get("subtype") != "success":
                        detail = event.get("result") or event.get("subtype")
                        raise ProviderError(f"claude: {detail}")
                    if not streamed and event.get("result"):
                        yield event["result"]
                    if model not in self.working_models:
                        self.working_models.add(model)
                        asyncio.create_task(self._replace(model))
                    return
        finally:
            _kill(proc)

    def status(self) -> dict:
        return {"spares": sorted(m for m, p in self.spares.items() if p.returncode is None)}


def _kill(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


def claude_event_text(event: object) -> str:
    if not isinstance(event, dict) or event.get("type") != "stream_event":
        return ""
    inner = event.get("event") or {}
    if inner.get("type") != "content_block_delta":
        return ""
    delta = inner.get("delta") or {}
    return delta.get("text", "") if delta.get("type") == "text_delta" else ""
