"""`q` — ask an LLM a quick question from the terminal."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable

from c5_ask import __version__, history
from c5_ask.providers import PROVIDERS, Message, ProviderError, stream_answer

MAX_PIPED_CHARS = 200_000

# Disable globbing for `q` so `?`, `*` and `[]` in questions reach us untouched.
SHELL_INIT = {
    "zsh": "alias q='noglob q'\n",
    # bash has no per-command noglob. An alias like `set -f; q` breaks pipes
    # (`cat log | q why` would lose stdin), so bash users quote questions with * ? [ instead.
    "bash": "# q: bash can't disable globbing per command; quote questions containing * ? [\n",
}

# Terminal control sequences in model output could set the window title, write the
# clipboard (OSC 52) or hide text. Keep printable text, newlines and tabs only.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    return CONTROL_CHARS.sub("", text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="q",
        description="Ask Claude, OpenAI or Gemini a quick question and get a concise answer.",
        epilog=(
            "examples:\n"
            "  q how do I parse json in go\n"
            "  q -c and with a struct\n"
            "  cat error.log | q why is this failing\n"
            "  q -p gemini regex for an ipv4 address\n"
            "  q -p codex explain git rebase --onto     (ChatGPT subscription, no API key)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--continue",
        dest="cont",
        action="store_true",
        help="continue the previous conversation",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=sorted(PROVIDERS),
        default=os.environ.get("C5_ASK_PROVIDER", "claude"),
        help="provider (default: $C5_ASK_PROVIDER or claude)",
    )
    parser.add_argument(
        "-m", "--model", help="model id (default: $C5_ASK_MODEL or the provider default)"
    )
    parser.add_argument(
        "-r", "--raw", action="store_true", help="print plain text instead of rendered Markdown"
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="print one JSON object (provider, model, question, answer)",
    )
    parser.add_argument("--clear", action="store_true", help="forget the previous conversation")
    parser.add_argument(
        "--init",
        choices=sorted(SHELL_INIT),
        metavar="SHELL",
        help="print shell setup for zsh or bash",
    )
    parser.add_argument(
        "--daemon",
        choices=["start", "stop", "restart", "status", "install", "uninstall", "run"],
        metavar="ACTION",
        help="manage the background service for codex/claude-code: "
        "start, stop, restart, status, install (start at login), uninstall, run",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    # REMAINDER: everything after the first word is the question, so `q what does ls -la do`
    # doesn't treat -la as an option.
    parser.add_argument("question", nargs=argparse.REMAINDER, help="your question")
    return parser


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.question[:1] == ["--"]:  # `q -- -v flag meaning` asks about "-v flag meaning"
        args.question = args.question[1:]
    return args


def compose_prompt(question: str, piped: str | None) -> str:
    if not piped:
        return question
    piped = piped.rstrip()
    if len(piped) > MAX_PIPED_CHARS:
        print(f"q: piped input truncated to {MAX_PIPED_CHARS:,} characters", file=sys.stderr)
        piped = piped[:MAX_PIPED_CHARS]
    if not question:
        return piped
    return f"{question}\n\n<input>\n{piped}\n</input>"


def read_piped_stdin() -> str | None:
    if sys.stdin is None or sys.stdin.isatty():
        return None
    # Bytes + replace: binary or non-UTF-8 input (`cat app.bin | q ...`) must not crash.
    # Read one byte past the cap so truncation is still detected without slurping it all.
    data = sys.stdin.buffer.read(MAX_PIPED_CHARS * 4 + 1)
    return data.decode("utf-8", "replace")


def render(chunks: Iterable[str], raw: bool) -> str:
    """Print chunks as they arrive and return the full answer (unsanitized, for history)."""
    answer = ""
    if raw or not sys.stdout.isatty():
        for chunk in chunks:
            answer += chunk
            sys.stdout.write(sanitize(chunk))
            sys.stdout.flush()
        if answer and not answer.endswith("\n"):
            sys.stdout.write("\n")
        return answer

    from rich.console import Console  # deferred: keeps `--raw` and pipes fast
    from rich.live import Live
    from rich.markdown import Markdown

    console = Console()
    # Transient live preview while streaming, then one clean render of the whole answer
    # (live regions can't redraw text that has already scrolled off-screen).
    with Live(
        console=console, transient=True, refresh_per_second=15, vertical_overflow="crop"
    ) as live:
        for chunk in chunks:
            answer += chunk
            live.update(_markdown(Markdown, answer))
    console.print(_markdown(Markdown, answer))
    return answer


def _markdown(markdown_cls, text: str):
    # hyperlinks=False prints link targets in full instead of hiding them behind the text.
    return markdown_cls(sanitize(text), hyperlinks=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.init:
        sys.stdout.write(SHELL_INIT[args.init])
        return 0
    if args.daemon:
        from c5_ask import daemon

        try:
            return daemon.control(args.daemon)
        except (ProviderError, OSError, subprocess.CalledProcessError) as exc:
            print(f"q: {exc}", file=sys.stderr)
            return 1
    if args.clear:
        history.clear()
        if not args.question:
            return 0

    prompt = compose_prompt(" ".join(args.question).strip(), read_piped_stdin())
    if not prompt.strip():
        build_parser().print_usage(sys.stderr)
        return 2

    model = args.model or os.environ.get("C5_ASK_MODEL") or PROVIDERS[args.provider].default_model
    messages: list[Message] = history.load() if args.cont else []
    messages.append({"role": "user", "content": prompt})

    try:
        chunks = stream_answer(args.provider, model, messages)
        if args.json:
            answer = "".join(chunks)
            payload = {
                "provider": args.provider,
                "model": model,
                "question": prompt,
                "answer": answer,
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            answer = render(chunks, args.raw)
    except ProviderError as exc:
        print(f"q: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # Reader went away (`q ... | head`): silence the final flush at exit, like other CLIs.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 141

    messages.append({"role": "assistant", "content": answer})
    try:
        history.save(messages)
    except OSError as exc:
        print(f"q: could not save history: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
