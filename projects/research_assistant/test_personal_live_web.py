from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api import create_app
from personal_live_cli import ModelConfig
from personal_live_web import (
    WEB_TRACE_PATH,
    build_personal_web_app,
    choose_available_port,
    find_personal_service,
    main,
)


class PersonalLiveWebTests(unittest.TestCase):
    def test_root_serves_personal_playground_without_loading_rag(self):
        app = create_app(strategy="lexical")
        client = TestClient(app)

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("岗位需求 RAG 分析助手", response.text)
        self.assertIn("personal-live-rag-ui", response.text)
        self.assertIn("fetch('/ask'", response.text)
        self.assertIsNone(app.state.service)

    def test_personal_web_factory_uses_full_real_model_chain(self):
        config = ModelConfig(
            api_key="test-secret",
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash",
            source="test",
        )
        factory = Mock(return_value=object())

        with patch.dict(os.environ, {}, clear=True):
            app = build_personal_web_app(config, app_factory=factory)
            self.assertEqual(os.environ["MODEL_NAME"], "deepseek-v4-flash")
            self.assertEqual(os.environ["MODEL_MAX_RETRIES"], "0")
            self.assertEqual(os.environ["MODEL_MAX_TOKENS"], "1500")

        self.assertIs(app, factory.return_value)
        factory.assert_called_once_with(
            strategy="hybrid_rerank",
            generator_mode="model",
            timeout_seconds=120.0,
            trace_path=WEB_TRACE_PATH,
            rerank_candidate_k=30,
        )

    def test_existing_personal_service_is_reused_without_loading_config(self):
        with (
            patch("personal_live_web._is_personal_service_running", return_value=True),
            patch("personal_live_web.resolve_model_config") as resolve_config,
        ):
            exit_code = main(["--no-browser"])

        self.assertEqual(exit_code, 0)
        resolve_config.assert_not_called()

    def test_busy_unrelated_port_moves_to_next_available_port(self):
        availability = {8000: False, 8001: False, 8002: True}

        selected = choose_available_port(
            8000,
            is_available=lambda port: availability.get(port, False),
            attempts=4,
        )

        self.assertEqual(selected, 8002)

    def test_shifted_personal_service_is_reused(self):
        services = {8000: False, 8001: False, 8002: True}

        selected = find_personal_service(
            8000,
            is_personal_service=lambda port: services.get(port, False),
            attempts=4,
        )

        self.assertEqual(selected, 8002)


if __name__ == "__main__":
    unittest.main()
