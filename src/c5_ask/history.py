"""Persist the most recent conversation so `q -c` can continue it."""

from __future__ import annotations

import json
import os
from pathlib import Path

from c5_ask.providers import Message


def history_path() -> Path:
    from c5_ask.daemon import state_dir  # private (0700, owned by the user) state directory

    return state_dir() / "last.json"


def load(path: Path | None = None) -> list[Message]:
    path = path or history_path()
    try:
        messages = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(messages, list):
        return []
    return [
        m
        for m in messages
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ]


def save(messages: list[Message], path: Path | None = None) -> None:
    path = path or history_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    # Piped files can be sensitive: keep the history private to the user.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(messages, f)
    tmp.replace(path)


def clear(path: Path | None = None) -> None:
    (path or history_path()).unlink(missing_ok=True)
