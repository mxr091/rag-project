"""Personal interactive entry point for the full local RAG + real-model chain.

The launcher deliberately discovers existing local configuration without ever
printing or copying the API key.  It keeps the embedding and reranker models in
one process so that a user can ask several questions without paying the model
loading cost for every turn.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_TRACE_PATH = PROJECT_ROOT / "logs" / "personal_live_trials.jsonl"
PERSONAL_MODEL_MAX_TOKENS = 1500
DEFAULT_ENV_FILES = (
    PROJECT_ROOT / ".env",
    WORKSPACE_ROOT
    / "projects"
    / "agent-reproduction-lab"
    / "reproductions"
    / "chapter4"
    / ".env",
)


@dataclass(frozen=True)
class ModelConfig:
    api_key: str = field(repr=False)
    endpoint: str
    model: str
    source: str

    def as_environment(self) -> dict[str, str]:
        return {
            "MODEL_API_KEY": self.api_key,
            "MODEL_ENDPOINT": self.endpoint,
            "MODEL_NAME": self.model,
        }


def read_env_file(path: Path) -> dict[str, str]:
    """Read a small dotenv file without interpolation or secret output."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name:
            continue
        values[name] = value.strip().strip('"').strip("'")
    return values


def _chat_completions_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _config_from_mapping(values: Mapping[str, str], source: str) -> ModelConfig | None:
    direct_key = values.get("MODEL_API_KEY", "").strip()
    direct_endpoint = values.get("MODEL_ENDPOINT", "").strip()
    direct_model = values.get("MODEL_NAME", "").strip()
    if direct_key and direct_endpoint and direct_model:
        return _validate_config(direct_key, direct_endpoint, direct_model, source)

    llm_key = values.get("LLM_API_KEY", "").strip()
    llm_base = values.get("LLM_BASE_URL", "").strip()
    llm_model = values.get("LLM_MODEL_ID", "").strip()
    if llm_key and llm_base and llm_model:
        return _validate_config(
            llm_key,
            _chat_completions_endpoint(llm_base),
            llm_model,
            source,
        )

    deepseek_key = values.get("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        deepseek_base = values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        deepseek_model = values.get("DEEPSEEK_MODEL", "deepseek-flash")
        return _validate_config(
            deepseek_key,
            _chat_completions_endpoint(deepseek_base),
            deepseek_model,
            source,
        )
    return None


def _validate_config(api_key: str, endpoint: str, model: str, source: str) -> ModelConfig:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"模型地址必须是无内嵌凭据的完整 HTTPS 地址：{source}")
    return ModelConfig(api_key=api_key, endpoint=endpoint, model=model, source=source)


def resolve_model_config(
    environment: Mapping[str, str] | None = None,
    env_files: Sequence[Path] = DEFAULT_ENV_FILES,
) -> ModelConfig:
    """Resolve process variables first, then known ignored local dotenv files."""
    process_values = os.environ if environment is None else environment
    process_config = _config_from_mapping(process_values, "当前进程环境变量")
    if process_config is not None:
        return process_config

    checked: list[str] = []
    for path in env_files:
        checked.append(str(path))
        if not path.is_file():
            continue
        config = _config_from_mapping(read_env_file(path), str(path))
        if config is not None:
            return config

    checked_text = "\n  - ".join(checked)
    raise RuntimeError(
        "没有找到可用的 DeepSeek 配置。已检查当前进程变量和以下本地文件：\n"
        f"  - {checked_text}\n"
        "需要 MODEL_*、LLM_* 或 DEEPSEEK_* 三组变量中的任意完整一组。"
    )


def configure_runtime(config: ModelConfig) -> None:
    os.environ.update(config.as_environment())
    # The personal entry point promises bounded paid calls.  Do not inherit a
    # more permissive retry/output policy from an unrelated parent process.
    os.environ["MODEL_MAX_RETRIES"] = "0"
    os.environ["MODEL_RETRY_BACKOFF_SECONDS"] = "0.5"
    os.environ["MODEL_MAX_TOKENS"] = str(PERSONAL_MODEL_MAX_TOKENS)

    model_cache = WORKSPACE_ROOT / "output" / "model-cache"
    if model_cache.is_dir():
        os.environ.setdefault("HF_HOME", str(model_cache))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def config_summary(config: ModelConfig) -> str:
    host = urlsplit(config.endpoint).hostname or "unknown"
    return (
        f"配置来源：{config.source}\n"
        f"模型：{config.model}\n"
        f"服务：{host}\n"
        "API Key：已读取（不会显示、不会写入项目文件）"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="个人真实岗位 RAG 交互测试")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查现有模型配置，不加载模型、不发起 API 请求",
    )
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--candidate-k", type=int, default=30)
    return parser


def _print_help() -> None:
    print("\n可以直接输入问题，例如：")
    print("  1. 哪些深圳岗位明确提到 LangGraph？请给出岗位和证据。")
    print("  2. 找出同时要求 RAG、Python 和部署能力的岗位，并比较要求。")
    print("  3. 哪些岗位更适合有 PyTorch 和 Agent 项目经验的候选人？只依据证据回答。")
    print("\n输入 help 再看示例；输入 exit 或 退出结束。")


def run_interactive(config: ModelConfig, *, top_k: int, candidate_k: int) -> int:
    from rag_observability import append_rag_trace
    from service_factory import create_job_service

    print("\n正在初始化完整链路：BM25 + BGE + RRF + ReRank + 真实大模型……")
    print("首次提问会加载本地模型，可能需要几十秒。")
    service = create_job_service(
        strategy="hybrid_rerank",
        generator_mode="model",
        rerank_candidate_k=candidate_k,
    )
    print("初始化完成。只有提交问题时才会调用外部模型并可能计费。")
    _print_help()

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已结束。")
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"} or question == "退出":
            print("已结束。")
            return 0
        if question.lower() in {"help", "?"} or question == "帮助":
            _print_help()
            continue

        try:
            result = service.answer(question, top_k=top_k)
            append_rag_trace(
                DEFAULT_TRACE_PATH,
                result,
                strategy="hybrid_rerank",
                top_k=top_k,
            )
        except Exception as error:  # Provider errors are already credential-safe.
            print(f"\n调用失败：{type(error).__name__}: {error}")
            print("你可以重试，或输入 exit 结束。")
            continue

        print(f"\n助手：{result.answer}")
        if result.citations:
            print("\n来源：")
            for citation in result.citations:
                suffix = f" {citation.source_url}" if citation.source_url else ""
                print(f"[{citation.number}] {citation.source}{suffix}")
        if result.refused:
            print(f"\n[安全拒答：{result.failure_type}]")
        usage = dict(result.model_usage) if result.model_usage else None
        print(
            f"\n[mode={result.answer_mode} strategy=hybrid_rerank "
            f"latency_ms={result.latency_ms:.1f} usage={usage}]"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.candidate_k <= 0:
        raise SystemExit("candidate-k must be positive")
    try:
        config = resolve_model_config()
        configure_runtime(config)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 2

    print("岗位需求 RAG 分析助手：个人真实模型测试")
    print(config_summary(config))
    print(
        "链路：hybrid_rerank + model；自动重试：0；"
        f"单次最多输出 {PERSONAL_MODEL_MAX_TOKENS} tokens"
    )
    if args.check_config:
        print("配置检查通过；未加载本地模型，未调用外部 API。")
        return 0
    return run_interactive(config, top_k=args.top_k, candidate_k=args.candidate_k)


if __name__ == "__main__":
    raise SystemExit(main())
