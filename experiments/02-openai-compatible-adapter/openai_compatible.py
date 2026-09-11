"""OpenAI-compatible HTTP adapter for the framework-free Agent loop.

The adapter is intentionally small: it translates the local message/tool
format into the common chat-completions shape and parses one final answer or
one tool call. The Agent loop remains independent from this transport layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-minimal-agent"))
from agent_loop import ModelResponse, ToolCall  # noqa: E402


class ModelAPIError(RuntimeError):
    """Raised when the provider cannot return a valid model response."""

    def __init__(self, message: str, *, failure_type: str = "model_api_error", usage=None):
        super().__init__(message)
        self.failure_type = failure_type
        self.usage = usage


@dataclass(frozen=True)
class OpenAICompatibleModel:
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0
    opener: Callable[..., Any] = urlopen
    max_retries: int = 0
    retry_backoff_seconds: float = 0.5
    sleeper: Callable[[float], None] = sleep
    max_tokens: int | None = None
    thinking_type: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.thinking_type not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled, disabled or None")

    def respond(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "tools": [self._provider_tool(tool) for tool in tools],
            "tool_choice": "auto",
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_type is not None:
            payload["thinking"] = {"type": self.thinking_type}
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                if attempt < self.max_retries and self._is_retryable_http_error(exc):
                    self._wait_before_retry(attempt)
                    continue
                raise ModelAPIError(f"model API returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise ModelAPIError(f"model API request failed: {exc}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModelAPIError("model API returned invalid JSON") from exc

        return self._parse_response(body)

    def _wait_before_retry(self, attempt: int) -> None:
        self.sleeper(self.retry_backoff_seconds * (2 ** attempt))

    @staticmethod
    def _is_retryable_http_error(error: HTTPError) -> bool:
        if 500 <= error.code < 600:
            return True
        if error.code != 429:
            return False
        try:
            payload = json.loads(error.read().decode("utf-8"))
            provider_code = payload.get("error", {}).get("code")
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return True
        return provider_code not in {"insufficient_quota", "credit_balance_exhausted"}

    @staticmethod
    def _provider_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }

    @staticmethod
    def _provider_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        call_number = 0
        for message in messages:
            role = message["role"]
            if role in {"user", "system"}:
                converted.append({"role": role, "content": message["content"]})
            elif role == "assistant" and "tool_call" in message:
                call_number += 1
                call = message["tool_call"]
                converted.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"local-call-{call_number}",
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
            elif role == "assistant":
                converted.append({"role": "assistant", "content": message["content"]})
            elif role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"local-call-{call_number}",
                        "name": message["name"],
                        "content": json.dumps(message["content"], ensure_ascii=False),
                    }
                )
            else:
                raise ModelAPIError(f"unsupported local message role: {role}")
        return converted

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> ModelResponse:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("model API response has no choices[0].message") from exc

        usage = OpenAICompatibleModel._parse_usage(body.get("usage"))
        if body["choices"][0].get("finish_reason") == "length":
            raise ModelAPIError("model output exceeded token limit", failure_type="output_truncated", usage=usage)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            try:
                function = tool_calls[0]["function"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                return ModelResponse(
                    tool_call=ToolCall(function["name"], arguments),
                    usage=usage,
                )
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                raise ModelAPIError("model API returned an invalid tool call") from exc

        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ModelAPIError("model API returned neither content nor a tool call")
        return ModelResponse(content=content, usage=usage)

    @staticmethod
    def _parse_usage(raw_usage: Any) -> dict[str, int] | None:
        if raw_usage is None:
            return None
        if not isinstance(raw_usage, dict):
            raise ModelAPIError("model API returned invalid usage metadata")
        usage: dict[str, int] = {}
        for key, value in raw_usage.items():
            if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
        return usage or None
