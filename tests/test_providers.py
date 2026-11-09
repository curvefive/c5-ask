import http.client
import io
import json
import urllib.error

import pytest

from c5_ask import providers
from c5_ask.providers import ProviderError, build_request, extract_text, iter_sse_data

MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "again"},
]


def test_claude_request_shape():
    url, headers, body = build_request("claude", "m", "k", MESSAGES)
    assert url.endswith("/v1/messages")
    assert headers["x-api-key"] == "k"
    assert body["system"] == providers.SYSTEM_PROMPT
    assert body["messages"] == MESSAGES and body["stream"] is True


def test_openai_request_prepends_system_message():
    _, headers, body = build_request("openai", "m", "k", MESSAGES)
    assert headers["authorization"] == "Bearer k"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1:] == MESSAGES


def test_gemini_request_maps_assistant_to_model():
    url, headers, body = build_request("gemini", "gm", "k", MESSAGES)
    assert "models/gm:streamGenerateContent?alt=sse" in url
    assert headers["x-goog-api-key"] == "k"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][1]["parts"] == [{"text": "hello"}]


def test_unknown_provider():
    with pytest.raises(ValueError):
        build_request("nope", "m", "k", MESSAGES)


@pytest.mark.parametrize(
    ("provider", "event", "text"),
    [
        ("claude", {"type": "content_block_delta", "delta": {"text": "a"}}, "a"),
        ("claude", {"type": "message_start", "message": {}}, ""),
        ("openai", {"choices": [{"delta": {"content": "b"}}]}, "b"),
        ("openai", {"choices": [{"delta": {"role": "assistant"}}]}, ""),
        ("openai", {"choices": []}, ""),
        ("gemini", {"candidates": [{"content": {"parts": [{"text": "c"}, {"text": "d"}]}}]}, "cd"),
        (
            "gemini",
            {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}]}}]},
            "",
        ),
        ("gemini", {"usageMetadata": {}}, ""),
    ],
)
def test_extract_text(provider, event, text):
    assert extract_text(provider, event) == text


def test_extract_text_raises_on_stream_error():
    with pytest.raises(ProviderError, match="overloaded"):
        extract_text("claude", {"type": "error", "error": {"message": "overloaded"}})


def test_iter_sse_data_skips_noise_and_stops_at_done():
    lines = [b"event: x\n", b'data: {"a":1}\n', b"\n", b": ping\n", b"data: [DONE]\n", b"data: 2\n"]
    assert list(iter_sse_data(iter(lines))) == ['{"a":1}']


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY.*-p claude-code"):
        list(providers.stream_answer("claude", "m", MESSAGES))


def test_gemini_accepts_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert providers.api_key_for("gemini") == "g"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_stream_answer_end_to_end(monkeypatch):
    events = [
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"text": "Hel"}},
        {"type": "content_block_delta", "delta": {"text": "lo"}},
        {"type": "message_stop"},
    ]
    raw = "".join(f"event: e\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    sent = {}

    def fake_urlopen(request, timeout):
        sent["body"] = json.loads(request.data)
        sent["key"] = request.get_header("X-api-key")
        return FakeResponse(raw)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    assert "".join(providers.stream_answer("claude", "m", MESSAGES)) == "Hello"
    assert sent["key"] == "secret" and sent["body"]["model"] == "m"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"error": {"message": "bad key"}}', "bad key"),
        (b'[{"error": {"message": "gemini says no"}}]', "gemini says no"),
        (b"plain failure", "plain failure"),
    ],
)
def test_http_error_message(monkeypatch, body, expected):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(body))

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError, match=f"openai HTTP 401: {expected}"):
        list(providers.stream_answer("openai", "m", MESSAGES))


def test_network_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError, match="could not reach openai: no route"):
        list(providers.stream_answer("openai", "m", MESSAGES))


@pytest.mark.parametrize(
    "model", ["claude-sonnet-5-5[1m]", "ft:gpt-4o:org:custom:id", "gemini-flash-latest"]
)
def test_valid_models(model):
    assert providers.validate_model(model) == model


@pytest.mark.parametrize(
    "model",
    ["--dangerously-skip-permissions", "a/b", "../x", "m?key=1", "a b", "", None, "x" * 200],
)
def test_invalid_models(model):
    with pytest.raises(ProviderError, match="invalid model"):
        providers.validate_model(model)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "x"}],
        [{"role": "user"}],
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        "x",
    ],
)
def test_invalid_messages(messages):
    with pytest.raises(ProviderError):
        providers.validate_messages(messages)


def test_connection_dropped_mid_stream(monkeypatch):
    class Dropping(FakeResponse):
        def __iter__(self):
            yield b'data: {"choices": [{"delta": {"content": "par"}}]}\n'
            raise http.client.IncompleteRead(b"")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda r, timeout: Dropping(b""))
    chunks = providers.stream_answer("openai", "m", MESSAGES)
    assert next(chunks) == "par"
    with pytest.raises(ProviderError, match="connection to openai failed"):
        next(chunks)


def test_malformed_event(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(
        providers.urllib.request, "urlopen", lambda r, timeout: FakeResponse(b"data: {nope\n")
    )
    with pytest.raises(ProviderError, match="unexpected response"):
        list(providers.stream_answer("openai", "m", MESSAGES))


def test_non_object_events_are_ignored():
    assert extract_text("openai", ["x"]) == ""
