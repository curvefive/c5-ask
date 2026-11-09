import json
import shutil
import stat
import subprocess
import sys

import pytest

from c5_ask import cli, history
from c5_ask.providers import ProviderError


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("C5_ASK_PROVIDER", raising=False)
    monkeypatch.delenv("C5_ASK_MODEL", raising=False)
    monkeypatch.setattr(cli, "read_piped_stdin", lambda: None)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_stream(provider, model, messages):
        seen.append((provider, model, [dict(m) for m in messages]))
        yield "ans"
        yield "wer"

    monkeypatch.setattr(cli, "stream_answer", fake_stream)
    return seen


def test_raw_answer_and_history(calls, capsys):
    assert cli.main(["-r", "what", "is", "x?"]) == 0
    assert capsys.readouterr().out == "answer\n"
    assert calls == [("claude", "claude-haiku-5-5", [{"role": "user", "content": "what is x?"}])]
    assert history.load() == [
        {"role": "user", "content": "what is x?"},
        {"role": "assistant", "content": "answer"},
    ]


def test_continue_sends_previous_turns(calls):
    cli.main(["-r", "first"])
    cli.main(["-r", "-c", "second"])
    sent = calls[1][2]
    assert [m["content"] for m in sent] == ["first", "answer", "second"]


def test_without_continue_starts_fresh(calls):
    cli.main(["-r", "first"])
    cli.main(["-r", "second"])
    assert len(calls[1][2]) == 1


def test_question_flags_are_not_parsed(calls):
    cli.main(["-r", "what", "does", "ls", "-la", "do"])
    assert calls[0][2][0]["content"] == "what does ls -la do"


def test_provider_and_model_selection(calls, monkeypatch):
    cli.main(["-r", "-p", "gemini", "x"])
    cli.main(["-r", "-p", "openai", "-m", "custom", "x"])
    monkeypatch.setenv("C5_ASK_PROVIDER", "openai")
    monkeypatch.setenv("C5_ASK_MODEL", "env-model")
    cli.main(["-r", "x"])
    assert [(p, m) for p, m, _ in calls] == [
        ("gemini", "gemini-flash-latest"),
        ("openai", "custom"),
        ("openai", "env-model"),
    ]


def test_json_output(calls, capsys):
    assert cli.main(["--json", "hi"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "provider": "claude",
        "model": "claude-haiku-5-5",
        "question": "hi",
        "answer": "answer",
    }


def test_piped_input_is_wrapped(calls, monkeypatch):
    monkeypatch.setattr(cli, "read_piped_stdin", lambda: "Traceback...\n")
    cli.main(["-r", "why"])
    assert calls[0][2][0]["content"] == "why\n\n<input>\nTraceback...\n</input>"


def test_piped_input_alone_is_the_question(calls, monkeypatch):
    monkeypatch.setattr(cli, "read_piped_stdin", lambda: "explain this\n")
    cli.main(["-r"])
    assert calls[0][2][0]["content"] == "explain this"


def test_piped_input_is_truncated(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_PIPED_CHARS", 5)
    assert cli.compose_prompt("q", "abcdefgh") == "q\n\n<input>\nabcde\n</input>"
    assert "truncated" in capsys.readouterr().err


def test_empty_question_is_usage_error(calls, capsys):
    assert cli.main([]) == 2
    assert "usage: q" in capsys.readouterr().err
    assert calls == []


def test_provider_error_exits_nonzero_and_keeps_history(monkeypatch, capsys):
    history.save([{"role": "user", "content": "old"}])

    def failing(*_):
        raise ProviderError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "stream_answer", failing)
    assert cli.main(["-r", "x"]) == 1
    assert capsys.readouterr().err == "q: boom\n"
    assert history.load() == [{"role": "user", "content": "old"}]


def test_clear(calls):
    cli.main(["-r", "x"])
    assert cli.main(["--clear"]) == 0
    assert history.load() == []


def test_history_is_private_and_tolerates_corruption(tmp_path):
    path = tmp_path / "h" / "last.json"
    history.save([{"role": "user", "content": "secret"}], path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.write_text("{not json")
    assert history.load(path) == []
    path.write_text(json.dumps([{"role": "system", "content": "x"}, "junk"]))
    assert history.load(path) == []


def test_rendered_markdown_path(calls, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert cli.main(["x"]) == 0
    assert "answer" in capsys.readouterr().out


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_shell_init(shell, tmp_path):
    exe = shutil.which(shell)
    if not exe:
        pytest.skip(f"{shell} not installed")
    # A stub `q` on PATH that echoes its args and stdin; a file named `x` matches the glob `?`.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "q"
    stub.write_text('#!/bin/sh\nprintf "%s|" "$@"; printf "stdin=%s\\n" "$(cat)"\n')
    stub.chmod(0o755)
    (tmp_path / "x").touch()
    # Non-interactive shells: bash needs expand_aliases, and both parse a -c string up front,
    # so calls go through eval to be parsed after the alias exists (as at a prompt).
    prelude = "shopt -s expand_aliases" if shell == "bash" else ""
    script = (
        f"{prelude}\n{cli.SHELL_INIT[shell]}\ncd {tmp_path}\n"
        "eval \"q is it '*' </dev/null\"\n"
        'eval "echo PIPED | q why ?"\n'
        "echo ?\n"
    )
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    flags = ["--norc"] if shell == "bash" else ["-f"]
    out = subprocess.run(
        [exe, *flags, "-c", script], capture_output=True, text=True, env=env, check=True
    ).stdout.splitlines()
    assert out[0] == "is|it|*|stdin="
    # Piped input must survive the shell setup; zsh also leaves `?` unexpanded.
    assert out[1] == ("why|?|stdin=PIPED" if shell == "zsh" else "why|x|stdin=PIPED")
    assert out[2] == "x"  # globbing still works for other commands


def test_closed_stdout_pipe_exits_quietly(tmp_path):
    # `q ... | head -1` with a long answer: no traceback, exit 141 like other CLIs.
    script = (
        "from c5_ask import cli\n"
        "cli.read_piped_stdin = lambda: None\n"
        "cli.stream_answer = lambda *a: iter(['line\\n'] * 100000)\n"
        "raise SystemExit(cli.main(['-r', 'x']))\n"
    )
    result = subprocess.run(
        f"'{sys.executable}' -c \"$SCRIPT\" | head -1",
        shell=True,
        capture_output=True,
        text=True,
        env={"SCRIPT": script, "XDG_STATE_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert result.stdout == "line\n"
    assert result.stderr == ""


def test_terminal_escapes_are_stripped(calls, monkeypatch, capsys):
    evil = "ok\x1b]52;c;cHduZWQ=\x07\x1b]0;PWNED\x07\x1b[31mred\rX\n\tend"
    monkeypatch.setattr(cli, "stream_answer", lambda *a: iter([evil]))
    cli.main(["-r", "x"])
    assert capsys.readouterr().out == "ok]52;c;cHduZWQ=]0;PWNED[31mredX\n\tend\n"
    # History keeps the exact answer; only display is sanitized.
    assert history.load()[-1]["content"] == evil


def test_markdown_render_strips_escapes_and_shows_link_targets(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        cli, "stream_answer", lambda *a: iter(["\x1b]52;c;eA==\x07[docs](https://evil.example/x)"])
    )
    cli.main(["x"])
    out = capsys.readouterr().out
    assert "\x1b]52" not in out and "\x1b]8;" not in out
    assert "https://evil.example/x" in out


def test_double_dash_separates_question(calls):
    cli.main(["-r", "--", "-v", "flag"])
    assert calls[0][2][0]["content"] == "-v flag"


def test_invalid_model_is_reported(capsys):
    assert cli.main(["-r", "-m=--evil", "x"]) == 1
    assert "invalid model id" in capsys.readouterr().err


def test_binary_piped_input_does_not_crash(tmp_path):
    script = "from c5_ask import cli\nprint(repr(cli.read_piped_stdin()))\n"
    result = subprocess.run(
        [sys.executable, "-c", script], input=b"ok\xff\xfe\x00bin", capture_output=True
    )
    assert result.returncode == 0
    assert result.stdout.decode().strip() == repr("ok��\x00bin")
