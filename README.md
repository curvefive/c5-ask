# c5-ask

CurveFive terminal helper. Type `q` followed by a question and get a short, to-the-point answer
from Claude, OpenAI or Gemini, rendered as Markdown in your terminal. Use an API key, or your
existing ChatGPT / Claude subscription through the `codex` and `claude` CLIs.

```console
$ q python requests post json with timeout
```

```python
import requests

r = requests.post("https://api.example.com/items", json={"name": "x"}, timeout=10)
r.raise_for_status()
```

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pipx).

```bash
uv tool install .            # from a checkout; installs the `q` command
uv tool update-shell         # once, if ~/.local/bin is not on your PATH
```

**zsh:** add one line to `~/.zshrc` so `?`, `*` and `[]` in questions are not treated as file
globs:

```bash
eval "$(q --init zsh)"
```

**bash:** nothing to add. bash can't turn globbing off for a single command without breaking
pipes (`cat log | q why`), so quote questions that contain `*`, `?` or `[` (see Quoting).

Then pick how to pay for answers:

| Provider | Uses | Setup |
| --- | --- | --- |
| `claude` (default) | Anthropic API | `export ANTHROPIC_API_KEY=...` |
| `openai` | OpenAI API | `export OPENAI_API_KEY=...` |
| `gemini` | Gemini API | `export GEMINI_API_KEY=...` (`GOOGLE_API_KEY` also works) |
| `codex` | ChatGPT subscription | `codex` CLI installed and logged in (`codex login`) |
| `claude-code` | Claude subscription | `claude` CLI installed and logged in |

To make a subscription the default, add e.g. `export C5_ASK_PROVIDER=codex` to your profile.

Open a new terminal and run `q --version` to check.

Upgrade after pulling changes with `uv tool install --reinstall . && q --daemon restart`;
remove with `q --daemon uninstall; q --daemon stop; uv tool uninstall c5-ask`.

## Usage

```bash
q how do I reverse a list in python        # ask
q -c and in place                          # follow up on the previous answer
cat error.log | q why is this failing      # include piped input
git diff | q write a commit message
q -p gemini regex for an ipv4 address      # pick a provider
q -p openai -m gpt-5.4 explain rust lifetimes
q -p codex explain git rebase --onto        # ChatGPT subscription, no API key
q -p claude-code kubectl get pod logs       # Claude subscription, no API key
q --json what port does postgres use | jq -r .answer
```

| Flag | Description |
| --- | --- |
| `-c`, `--continue` | Continue the previous conversation instead of starting a new one. |
| `-p`, `--provider` | `claude`, `openai` or `gemini`. |
| `-m`, `--model` | Model id, overriding the provider default. |
| `-r`, `--raw` | Plain text instead of rendered Markdown. |
| `-j`, `--json` | One JSON object: `provider`, `model`, `question`, `answer`. |
| `--clear` | Forget the previous conversation. |
| `--init SHELL` | Print shell setup (`zsh`; `bash` needs none). |
| `--daemon ACTION` | Manage the subscription service: `start`, `stop`, `restart`, `status`, `install`, `uninstall`. |
| `-V`, `--version` | Print the version. |

Everything after the first word of the question is passed through as-is, so
`q what does ls -la do` works.

Output is rendered Markdown in a terminal and plain text when piped (`q ... > notes.md`).

### Quoting

The shell still interprets quotes, `;`, `|`, `&`, `>` and `$` before `q` sees them (and, in
bash, `*`, `?` and `[`). Wrap those questions in quotes:

```bash
q "what's the difference between && and ;"
q "what does ls *.py match?"     # needed in bash; zsh with --init handles it
```

Use `--` to ask about something that starts with a dash: `q -- -v flag of grep`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `C5_ASK_PROVIDER` | `claude` | Default provider. |
| `C5_ASK_MODEL` | provider default | Default model for every provider. Prefer `-m` if you switch providers. |
| `ANTHROPIC_API_KEY` | | Key for `claude`. |
| `OPENAI_API_KEY` | | Key for `openai`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | | Key for `gemini`. |
| `XDG_STATE_HOME` | `~/.local/state` | Where the last conversation is stored. |

Default models are chosen for speed: `claude-haiku-5-5`, `gpt-5.4-mini`,
`gemini-flash-latest`, `gpt-5.6-luna` (codex) and `claude-haiku-5-5` (claude-code).

## Subscription providers and the background service

`codex` and `claude-code` don't call an API directly. They drive the official `codex` and
`claude` CLIs, which are logged in with your subscription. Those CLIs take 3–8 seconds to
start, so `q` keeps them warm in a small background service:

- **codex:** one long-running `codex app-server`; every question gets a fresh, ephemeral thread.
- **claude-code:** one pre-started `claude -p` process waits for the next question; when it is
  used, a replacement starts in the background. Each process answers exactly one question, so
  questions never share context.

The service starts automatically on the first subscription question (that one is slow) and
keeps running until you stop it or log out. Typical answers then take 2–3 seconds end to end.

```bash
q --daemon status      # pid, uptime, warm processes
q --daemon stop
q --daemon install     # macOS: start at login via launchd (q --daemon uninstall to undo)
```

Both CLIs run in an empty working directory with no tools at all (see Security), and skip your
hooks, MCP servers, plugins and custom `model_provider`, so nothing changes the answer style.
Your own CLI configs are not modified.

Defaults: `gpt-5.6-luna` for codex (the models your account can use are listed by
`codex` → `/model`) and `claude-haiku-5-5` for claude-code. Override with `-m`.

The service listens on a Unix socket in a directory only you can access
(`~/.local/state/c5-ask/daemon.sock`) and logs to `~/.local/state/c5-ask/daemon.log` (the log
never records questions or answers, and is reset at 5 MB). After upgrading to a new version, an older
running service is replaced automatically on the next question.
Keep it for personal use: the subscriptions are licensed to you, not to a shared service.

## Security

`q` is built for piping untrusted text (logs, diffs, files) into a model, so it assumes that text
may contain prompt injections:

- **No tools for the model.** `claude` runs with `--tools ""` and no MCP servers. `codex` runs
  with its shell, exec, browser, computer-use, apps, plugins, web search, MCP and multi-agent
  features disabled, in a read-only sandbox, with every approval request refused. If `q` can't
  confirm the installed codex supports disabling its shell, it refuses to run codex. An
  injected instruction can change the answer text, but can't read files, run commands or
  reach the network.
- **Safe terminal output.** Control characters and escape sequences in answers are stripped
  before printing (no window-title or clipboard writes via OSC 52, no hidden text), and links
  are shown with their full URL. `--json` output is unmodified data.
- **Private state.** `~/.local/state/c5-ask` is created `0700`; `q` refuses to use it if it is
  not a directory owned by you. History is `0600`. The daemon socket is reachable only by you.
- **Validated input.** Model ids are restricted to safe characters (they become a CLI argument
  and a URL path), and daemon requests are validated before use.
- **Keys stay in your environment.** API keys are read from environment variables, sent only
  to the provider's HTTPS endpoint, and never logged or stored.

Remember that everything you ask, including piped input, is sent to the provider you choose.

## Conversation memory

Only the latest conversation is kept, at `$XDG_STATE_HOME/c5-ask/last.json` (mode `0600`,
since it can contain piped files). A plain `q` starts a new conversation and replaces it;
`q -c` appends to it; `q --clear` deletes it.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/ruff format --check . && .venv/bin/ruff check .
.venv/bin/pytest
```

The HTTP clients and the daemon use only the standard library; `rich` is the single runtime
dependency (Markdown rendering) and is only imported when output goes to a terminal.
Daemon tests use fake `codex`/`claude` executables that speak the real protocols, so CI needs
neither CLI.

| Module | Role |
| --- | --- |
| `cli.py` | Argument parsing, piped input, rendering, history. |
| `providers.py` | Streaming HTTP clients for the API providers. |
| `daemon.py` | Background service, its Unix-socket client, launchd install. |
| `backends.py` | Drives `codex app-server` and `claude -p` inside the daemon. |
