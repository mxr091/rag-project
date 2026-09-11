"""Utilities for turning AgentResult messages into inspectable run traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_loop import AgentResult


def build_trace(result: AgentResult) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for message in result.messages:
        if message["role"] == "user":
            events.append({"type": "user", "content": message["content"]})
        elif message["role"] == "assistant" and "tool_call" in message:
            events.append({"type": "tool_call", **message["tool_call"]})
        elif message["role"] == "tool":
            content = message["content"]
            events.append(
                {
                    "type": "tool_result",
                    "name": message["name"],
                    "ok": content.get("ok", False),
                    "content": content,
                }
            )
        elif message["role"] == "assistant":
            events.append({"type": "final_answer", "content": message["content"]})
    return events


def save_trace(result: AgentResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_trace(result), ensure_ascii=False, indent=2), encoding="utf-8")
