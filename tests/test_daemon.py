"""Daemon tests against fake `codex` and `claude` CLIs that speak the real protocols."""

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from c5_ask import backends, cli, daemon
from c5_ask.providers import ProviderError

FAKE_CODEX = """
import json, sys

def out(obj):
    print(json.dumps(obj), flush=True)

if sys.argv[1:] == ["features", "list"]:
    import os
    print(os.environ.get("FAKE_CODEX_FEATURES",
          "shell_tool  stable  true\\nweb_search_cached  removed  false\\napps  stable  true"))
    sys.exit(0)
# Every tool feature the installed codex knows must be disabled; removed ones must not be passed.
args = sys.argv[1:]
assert args[0] == "app-server", args
disabled = {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
assert disabled == {"shell_tool", "apps"}, disabled
assert 'web_search="disabled"' in args and "mcp_servers={}" in args, args

threads = 0
for line in sys.stdin:
    msg = json.loads(line)
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        out({"id": rid, "result": {}})
    elif method == "thread/start":
        threads += 1
        assert msg["params"]["sandbox"] == "read-only"
        out({"id": rid, "result": {"thread": {"id": f"t{threads}"}}})
    elif method == "turn/start":
        tid, text = msg["params"]["threadId"], msg["params"]["input"][0]["text"]
        out({"id": rid, "result": {"turn": {"id": "u1"}}})
        if "FAIL" in text:
            err = json.dumps({"error": {"message": "model not supported"}})
            turn = {"status": "failed", "error": {"message": err}}
            out({"method": "turn/completed", "params": {"threadId": tid, "turn": turn}})
            continue
        # A server->client approval request must be refused without breaking the turn.
        out({"id": 99, "method": "item/commandExecution/requestApproval", "params": {}})
        tag = "prev+" if "<previous_conversation>" in text else ""
        for delta in ("codex:", tag + text.splitlines()[-1]):
            out({"method": "item/agentMessage/delta", "params": {"threadId": tid, "delta": delta}})
        done = {"threadId": tid, "turn": {"status": "completed"}}
        out({"method": "turn/completed", "params": done})
"""

FAKE_CLAUDE = """
import json, sys

def out(obj):
    print(json.dumps(obj), flush=True)

args = sys.argv[1:]
assert args[args.index("--tools") + 1] == ""
model = args[args.index("--model") + 1]
out({"type": "system", "subtype": "init"})
text = json.loads(sys.stdin.readline())["message"]["content"]
if "HANG" in text:
    import os, pathlib, time
    pathlib.Path(text.split()[-1]).write_text(str(os.getpid()))
    out({"type": "stream_event", "event": {"type": "content_block_delta",
         "delta": {"type": "text_delta", "text": "partial"}}})
    time.sleep(60)
if "FAIL" in text:
    out({"type": "result", "subtype": "error_during_execution", "is_error": True,
         "result": "rate limited"})
else:
    tag = "prev+" if "<previous_conversation>" in text else ""
    for chunk in (f"claude[{model}]:", tag + text.splitlines()[-1]):
        out({"type": "stream_event", "event": {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": chunk}}})
    out({"type": "result", "subtype": "success", "is_error": False, "result": "ignored"})
sys.stdin.read()  # stay alive like the real CLI until stdin closes
"""


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("codex", FAKE_CODEX), ("claude", FAKE_CLAUDE)):
        script = bin_dir / name
        script.write_text(f"#!{sys.executable}\n{textwrap.dedent(body)}")
        script.chmod(0o755)
    # Only the fakes + system dirs: the real codex/claude must never be reachable from tests.
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("C5_ASK_MODEL", raising=False)
    monkeypatch.setattr(cli, "read_piped_stdin", lambda: None)
    yield tmp_path
    daemon.control("stop")


def ask(capsys, *argv):
    code = cli.main(["-r", *argv])
    out = capsys.readouterr()
    return code, out.out.strip(), out.err.strip()


def test_codex_and_claude_through_daemon(fake_env, capsys):
    assert ask(capsys, "-p", "codex", "hello") == (0, "codex:hello", "")
    assert ask(capsys, "-p", "codex", "-c", "again") == (0, "codex:prev+again", "")
    assert ask(capsys, "-p", "claude-code", "hi") == (0, "claude[claude-haiku-5-5]:hi", "")
    # Each claude question gets a fresh process, but -c still carries the conversation.
    assert ask(capsys, "-p", "claude-code", "-c", "more") == (
        0,
        "claude[claude-haiku-5-5]:prev+more",
        "",
    )
    assert ask(capsys, "-p", "claude-code", "-m", "claude-x", "m") == (0, "claude[claude-x]:m", "")

    assert daemon.control("status") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["codex"]["running"] is True
    assert "claude-haiku-5-5" in status["claude-code"]["spares"]


def test_backend_errors_are_reported(fake_env, capsys):
    code, _, err = ask(capsys, "-p", "codex", "FAIL")
    assert (code, err) == (1, "q: codex: model not supported")
    code, _, err = ask(capsys, "-p", "claude-code", "FAIL")
    assert (code, err) == (1, "q: claude: rate limited")
    # The daemon survives failed requests.
    assert ask(capsys, "-p", "codex", "ok")[0] == 0


def test_missing_cli(fake_env, capsys):
    (fake_env / "bin" / "codex").unlink()
    code, _, err = ask(capsys, "-p", "codex", "hi")
    assert code == 1 and "`codex` is not installed" in err


def test_stop_and_status_when_not_running(fake_env, capsys):
    assert daemon.control("status") == 1
    assert daemon.control("stop") == 0
    assert "not running" in capsys.readouterr().out


def test_long_state_dir_uses_short_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ("x" * 120)))
    path = daemon.socket_path()
    assert len(os.fsencode(path)) < 104 and path.parent.name == f"c5-ask-{os.getuid()}"
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_to_prompt():
    assert backends.to_prompt([{"role": "user", "content": "q"}]) == "q"
    prompt = backends.to_prompt(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert "[user]\na\n\n[assistant]\nb" in prompt and prompt.endswith("Follow-up question:\nc")


def test_rpc_error_unwraps_upstream_json():
    assert backends._rpc_error({"message": '{"detail": "nope"}'}) == "codex: nope"
    assert backends._rpc_error({"message": "plain"}) == "codex: plain"
    assert backends._rpc_error({}) == "codex: unknown error"


def test_find_cli_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(ProviderError):
        backends.find_cli("codex")


def raw_request(obj: dict) -> list[dict]:
    with daemon._connect(10) as sock, sock.makefile("rwb") as f:
        f.write(json.dumps(obj).encode() + b"\n")
        f.flush()
        return [json.loads(line) for line in f]


@pytest.mark.parametrize(
    "request_",
    [
        {
            "op": "ask",
            "provider": "claude-code",
            "model": "--dangerously-skip-permissions",
            "messages": [{"role": "user", "content": "x"}],
        },
        {
            "op": "ask",
            "provider": "shell",
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
        },
        {"op": "ask", "provider": "codex", "model": "m", "messages": [{"role": "system"}]},
        {"op": "nope"},
        ["not", "an", "object"],
    ],
)
def test_daemon_rejects_bad_requests(fake_env, request_):
    daemon.ensure_running()
    reply = raw_request(request_)
    assert len(reply) == 1 and "error" in reply[0]
    assert daemon.is_running()


def test_client_disconnect_kills_claude_process(fake_env):
    daemon.ensure_running()
    pid_file = fake_env / "claude.pid"
    request = {
        "op": "ask",
        "provider": "claude-code",
        "model": "claude-haiku-5-5",
        "messages": [{"role": "user", "content": f"HANG {pid_file}"}],
    }
    with daemon._connect(10) as sock, sock.makefile("rwb") as f:
        f.write(json.dumps(request).encode() + b"\n")
        f.flush()
        assert json.loads(f.readline()) == {"delta": "partial"}
    # The daemon notices on its next write; give it a moment, then the process must be gone.
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.1)
    assert not _alive(pid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Killed but not yet reaped shows as a zombie; that counts as gone.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.stdout.strip()[:1] not in ("", "Z")


def test_spare_processes_are_capped(fake_env, capsys):
    for model in ("m-one", "m-two", "m-three"):
        assert ask(capsys, "-p", "claude-code", "-m", model, "x")[0] == 0
    time.sleep(1)  # replacement spares start in the background
    daemon.control("status")
    spares = json.loads(capsys.readouterr().out)["claude-code"]["spares"]
    assert len(spares) <= backends.MAX_SPARE_MODELS and "m-three" in spares


def test_codex_refuses_to_run_if_tools_cannot_be_disabled(fake_env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_CODEX_FEATURES", "apps stable true")
    code, _, err = ask(capsys, "-p", "codex", "hi")
    assert code == 1 and "could not verify codex tools can be disabled" in err


def test_outdated_daemon_is_replaced(fake_env, monkeypatch):
    daemon.ensure_running()
    old_pid = daemon.daemon_status()["pid"]
    monkeypatch.setattr(daemon, "__version__", "0.0.0-new")  # as if q was just upgraded
    daemon.ensure_running()
    assert daemon.daemon_status()["pid"] != old_pid


def test_private_dir_fixes_permissions_and_rejects_symlinks(tmp_path):
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    daemon.private_dir(loose)
    assert loose.stat().st_mode & 0o777 == 0o700
    (tmp_path / "link").symlink_to(loose)
    with pytest.raises(ProviderError, match="refusing"):
        daemon.private_dir(tmp_path / "link")


def test_failing_model_gets_no_spare(fake_env, capsys):
    assert ask(capsys, "-p", "claude-code", "-m", "bad-model", "FAIL")[0] == 1
    time.sleep(0.5)
    daemon.control("status")
    assert "bad-model" not in json.loads(capsys.readouterr().out)["claude-code"]["spares"]
