"""A small, framework-free tool-calling Agent loop.

The model is deliberately injected as a Python object. This keeps the core
loop testable without API keys and makes the model provider replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class Model(Protocol):
    def respond(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> "ModelResponse":
        """Return either a final answer or one tool call."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    tool_call: ToolCall | None = None
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if bool(self.content) == (self.tool_call is not None):
            raise ValueError("response must contain exactly one of content or tool_call")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolError(Exception):
    pass


class MaxStepsExceeded(Exception):
    pass


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        self._tools = {tool.name: tool for tool in tools}

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._tools.values()]

    def invoke(self, call: ToolCall) -> Any:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"tool is not allowed: {call.name}")
        self._validate_arguments(tool, call.arguments)
        try:
            return tool.handler(**call.arguments)
        except Exception as exc:  # Convert tool failures into model-visible errors.
            raise ToolError(f"tool failed: {call.name}: {exc}") from exc

    @staticmethod
    def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolError("tool arguments must be an object")

        schema = tool.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = [name for name in required if name not in arguments]
        unknown = [name for name in arguments if name not in properties]
        if missing:
            raise ToolError(f"missing required arguments: {', '.join(missing)}")
        if unknown:
            raise ToolError(f"unknown arguments: {', '.join(unknown)}")

        for name, value in arguments.items():
            expected = properties[name].get("type")
            valid = {
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
            }.get(expected, True)
            if not valid:
                raise ToolError(f"argument {name!r} must be {expected}")
            rule = properties[name]
            if "enum" in rule and value not in rule["enum"]:
                raise ToolError(f"argument {name!r} is not an allowed value")
            if isinstance(value, str):
                if len(value) < rule.get("minLength", 0):
                    raise ToolError(f"argument {name!r} is too short")
                if "maxLength" in rule and len(value) > rule["maxLength"]:
                    raise ToolError(f"argument {name!r} is too long")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in rule and value < rule["minimum"]:
                    raise ToolError(f"argument {name!r} is below minimum")
                if "maximum" in rule and value > rule["maximum"]:
                    raise ToolError(f"argument {name!r} is above maximum")


@dataclass
class AgentResult:
    answer: str
    steps: int
    messages: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(self, model: Model, registry: ToolRegistry, max_steps: int = 5) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.model = model
        self.registry = registry
        self.max_steps = max_steps

    def run(self, user_input: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
        unresolved_tool_failure = False

        for step in range(1, self.max_steps + 1):
            response = self.model.respond(messages, self.registry.definitions())

            if response.tool_call is None:
                answer = response.content
                if unresolved_tool_failure:
                    answer = "工具执行失败，无法依据工具结果回答。"
                messages.append({"role": "assistant", "content": answer})
                return AgentResult(answer, step, messages)

            call = response.tool_call
            messages.append(
                {
                    "role": "assistant",
                    "tool_call": {"name": call.name, "arguments": call.arguments},
                }
            )
            try:
                result = self.registry.invoke(call)
                tool_message = {"ok": True, "result": result}
                unresolved_tool_failure = False
            except ToolError as exc:
                tool_message = {"ok": False, "error": str(exc)}
                unresolved_tool_failure = True

            messages.append({"role": "tool", "name": call.name, "content": tool_message})

        raise MaxStepsExceeded(f"agent exceeded max_steps={self.max_steps}")

