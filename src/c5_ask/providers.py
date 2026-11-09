"""Streaming chat clients for Anthropic, OpenAI and Gemini, using only the stdlib."""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}

SYSTEM_PROMPT = (
    "You are a terse terminal assistant for an experienced software developer. "
    "Answer only the question asked. Lead with the code, command or fact. "
    "No preamble, no restating the question, no caveats, no closing remarks. "
    "Prefer one short example over explanation. Keep answers under ~15 lines unless the "
    "user explicitly asks for more. Format with Markdown and fenced code blocks with a language."
)

# Model ids end up in a URL path (gemini) and a CLI argument (claude), so only allow the
# characters real ids use: no leading "-", no "/", "?", "#", "..", whitespace.
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@\[\]-]{0,127}")

MAX_TOKENS = 2048
TIMEOUT_SECONDS = 90


class ProviderError(RuntimeError):
    """A provider request failed; the message is safe to show to the user."""


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str | None  # None: uses the CLI's own subscription login via the daemon
    default_model: str


PROVIDERS: dict[str, Provider] = {
    "claude": Provider("claude", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
    "openai": Provider("openai", "OPENAI_API_KEY", "gpt-5.4-mini"),
    "gemini": Provider("gemini", "GEMINI_API_KEY", "gemini-flash-latest"),
    "codex": Provider("codex", None, "gpt-5.6-luna"),
    "claude-code": Provider("claude-code", None, "claude-haiku-5-5"),
}


def validate_model(model: object) -> str:
    if not isinstance(model, str) or not MODEL_RE.fullmatch(model) or ".." in model:
        raise ProviderError(f"invalid model id: {model!r}")
    return model


def validate_messages(messages: object) -> list[Message]:
    ok = (
        isinstance(messages, list)
        and messages
        and all(
            isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            for m in messages
        )
        and messages[-1]["role"] == "user"
    )
    if not ok:
        raise ProviderError("invalid conversation")
    return messages  # type: ignore[return-value]


def build_request(
    provider: str, model: str, api_key: str, messages: list[Message]
) -> tuple[str, dict[str, str], dict]:
    """Return (url, headers, json_body) for a streaming request."""
    if provider == "claude":
        return (
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "stream": True,
            },
        )
    if provider == "openai":
        return (
            "https://api.openai.com/v1/chat/completions",
            {"authorization": f"Bearer {api_key}"},
            {
                "model": model,
                "stream": True,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            },
        )
    if provider == "gemini":
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            ":streamGenerateContent?alt=sse",
            {"x-goog-api-key": api_key},
            {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [
                    {
                        "role": "model" if m["role"] == "assistant" else "user",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in messages
                ],
                "generationConfig": {"maxOutputTokens": MAX_TOKENS},
            },
        )
    raise ValueError(f"unknown provider: {provider}")


def extract_text(provider: str, event: dict) -> str:
    """Pull the text delta out of one decoded SSE event (raises on in-stream errors)."""
    if not isinstance(event, dict):
        return ""
    if "error" in event:
        err = event["error"]
        raise ProviderError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    if provider == "claude":
        if event.get("type") == "content_block_delta":
            return event.get("delta", {}).get("text", "")
        return ""
    if provider == "openai":
        choices = event.get("choices") or [{}]
        return choices[0].get("delta", {}).get("content") or ""
    candidates = event.get("candidates") or [{}]
    parts = candidates[0].get("content", {}).get("parts", [])
    # Gemini 3 marks hidden reasoning parts with "thought": true; never show those.
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def iter_sse_data(lines: Iterator[bytes]) -> Iterator[str]:
    """Yield the payload of each `data:` line in a server-sent-event stream."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield data


def api_key_for(provider: str) -> str:
    env = PROVIDERS[provider].key_env
    if env is None:
        raise ValueError(f"{provider} does not use an API key")
    key = os.environ.get(env, "").strip()
    if not key and provider == "gemini":
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise ProviderError(f"{env} is not set. Export it in your shell profile.")
    return key


def stream_answer(provider: str, model: str, messages: list[Message]) -> Iterator[str]:
    """Stream answer text chunks from the provider."""
    validate_model(model)
    validate_messages(messages)
    if PROVIDERS[provider].key_env is None:
        from c5_ask.daemon import stream_via_daemon

        yield from stream_via_daemon(provider, model, messages)
        return
    url, headers, body = build_request(provider, model, api_key_for(provider), messages)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "user-agent": "c5-ask", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            for data in iter_sse_data(response):
                text = extract_text(provider, json.loads(data))
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"{provider} HTTP {exc.code}: {_error_detail(exc)}") from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"could not reach {provider}: {exc.reason}") from None
    except TimeoutError:
        raise ProviderError(f"{provider} timed out after {TIMEOUT_SECONDS}s") from None
    except (OSError, http.client.HTTPException) as exc:  # connection dropped mid-answer
        raise ProviderError(f"connection to {provider} failed: {exc!r}") from None
    except ValueError as exc:  # malformed event from the server
        raise ProviderError(f"unexpected response from {provider}: {exc}") from None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", "replace")
    try:
        payload = json.loads(body)
        payload = payload[0] if isinstance(payload, list) else payload  # Gemini wraps in a list
        err = payload.get("error", payload)
        detail = err.get("message", body) if isinstance(err, dict) else str(err)
        return str(detail)[:500]
    except (ValueError, AttributeError, IndexError):
        return body[:500] or exc.reason
