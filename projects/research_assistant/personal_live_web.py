"""Launch the personal browser UI with the full RAG + real-model chain."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import urlopen

from personal_live_cli import (
    ModelConfig,
    PERSONAL_MODEL_MAX_TOKENS,
    PROJECT_ROOT,
    config_summary,
    configure_runtime,
    resolve_model_config,
)


WEB_TRACE_PATH = PROJECT_ROOT / "logs" / "personal_live_web_trials.jsonl"
PERSONAL_UI_MARKER = "personal-live-rag-ui"


def build_personal_web_app(
    config: ModelConfig,
    *,
    app_factory: Callable[..., Any] | None = None,
) -> Any:
    configure_runtime(config)
    if app_factory is None:
        from api import create_app

        app_factory = create_app
    return app_factory(
        strategy="hybrid_rerank",
        generator_mode="model",
        timeout_seconds=120.0,
        trace_path=WEB_TRACE_PATH,
        rerank_candidate_k=30,
    )


def _open_browser_when_ready(url: str) -> None:
    health_url = url.rstrip("/") + "/health"
    for _ in range(80):
        try:
            with urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(0.125)


def _is_personal_service_running(port: int) -> bool:
    base_url = f"http://127.0.0.1:{port}"
    try:
        with urlopen(base_url + "/health", timeout=0.5) as response:
            health = json.loads(response.read().decode("utf-8"))
        with urlopen(base_url + "/", timeout=0.5) as response:
            page = response.read().decode("utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        health.get("status") == "ok"
        and health.get("retrieval_strategy") == "hybrid_rerank"
        and PERSONAL_UI_MARKER in page
    )


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def find_personal_service(
    preferred: int,
    *,
    is_personal_service: Callable[[int], bool] | None = None,
    attempts: int = 20,
) -> int | None:
    checker = is_personal_service or _is_personal_service_running
    for port in range(preferred, min(preferred + attempts, 65536)):
        if checker(port):
            return port
    return None


def choose_available_port(
    preferred: int,
    *,
    is_available: Callable[[int], bool] = _port_available,
    attempts: int = 20,
) -> int:
    for port in range(preferred, min(preferred + attempts, 65536)):
        if is_available(port):
            return port
    raise RuntimeError(f"从 {preferred} 开始没有找到可用的本机端口")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动个人真实岗位 RAG 网页")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动服务但不自动打开浏览器，供诊断使用",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查既有模型配置，不启动服务、不调用 API",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        print("启动失败：port 必须在 1 到 65535 之间。", file=sys.stderr)
        return 2
    existing_port = None if args.check_config else find_personal_service(args.port)
    if existing_port is not None:
        existing_url = f"http://127.0.0.1:{existing_port}/"
        print(f"岗位需求 RAG 网页已经运行：{existing_url}")
        print("本次不会重复启动服务。")
        if not args.no_browser:
            webbrowser.open(existing_url)
        return 0

    try:
        config = resolve_model_config()
        selected_port = args.port if args.check_config else choose_available_port(args.port)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 2

    print("岗位需求 RAG 分析助手：个人真实模型网页")
    print(config_summary(config))
    print(
        "链路：BM25 + BGE + RRF + ReRank + 真实模型；自动重试：0；"
        f"单次最多输出 {PERSONAL_MODEL_MAX_TOKENS} tokens"
    )
    if args.check_config:
        print("配置检查通过；未启动服务、未加载模型、未调用外部 API。")
        return 0

    if selected_port != args.port:
        print(f"端口 {args.port} 已被其他程序占用，自动改用 {selected_port}。")
    app = build_personal_web_app(config)
    url = f"http://127.0.0.1:{selected_port}/"
    print(f"网页地址：{url}")
    print("提交普通证据问题时才会调用外部模型并可能计费。")
    print("关闭此窗口或按 Ctrl+C 可停止服务。")

    if not args.no_browser:
        opener = threading.Thread(
            target=_open_browser_when_ready,
            args=(url,),
            daemon=True,
        )
        opener.start()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=selected_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
