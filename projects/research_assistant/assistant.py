"""Agent-facing orchestration for the research assistant MVP."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "01-minimal-agent"))
from agent_loop import Agent, AgentResult, Model, Tool, ToolRegistry  # noqa: E402

from retriever import InMemoryRetriever


RESEARCH_INSTRUCTIONS = """你是科研资料助手。请先使用 search_documents 检索资料，再只根据检索结果回答。
回答中的关键结论必须标注来源文件名；如果检索结果没有足够证据，明确说明资料不足，不要编造内容。"""


def make_search_tool(retriever: InMemoryRetriever) -> Tool:
    def search_documents(query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return [
            {
                "source": result.chunk.source,
                "text": result.chunk.text,
                "score": round(result.score, 4),
            }
            for result in retriever.search(query, top_k=top_k)
        ]

    return Tool(
        name="search_documents",
        description="Search local research materials and return evidence snippets.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        handler=search_documents,
    )


class ResearchAssistant:
    def __init__(self, model: Model, retriever: InMemoryRetriever, max_steps: int = 3) -> None:
        self.agent = Agent(model, ToolRegistry([make_search_tool(retriever)]), max_steps=max_steps)

    def answer(self, question: str) -> AgentResult:
        prompt = f"{RESEARCH_INSTRUCTIONS}\n\n用户问题：{question}"
        return self.agent.run(prompt)
