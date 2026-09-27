from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_live_cli import (
    ModelConfig,
    config_summary,
    configure_runtime,
    read_env_file,
    resolve_model_config,
)


class PersonalLiveCliTests(unittest.TestCase):
    def test_existing_llm_dotenv_is_mapped_to_main_project_variables(self):
        existing_file = Path(__file__)
        with patch(
            "personal_live_cli.read_env_file",
            return_value={
                "LLM_API_KEY": "fake-key",
                "LLM_BASE_URL": "https://api.deepseek.com",
                "LLM_MODEL_ID": "deepseek-v4-flash",
            },
        ):
            config = resolve_model_config(environment={}, env_files=(existing_file,))

        self.assertEqual(config.endpoint, "https://api.deepseek.com/chat/completions")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(config.as_environment()["MODEL_API_KEY"], "fake-key")

    def test_complete_process_model_config_takes_precedence(self):
        config = resolve_model_config(
            environment={
                "MODEL_API_KEY": "fake-key",
                "MODEL_ENDPOINT": "https://provider.example/v1/chat/completions",
                "MODEL_NAME": "test-model",
            },
            env_files=(),
        )

        self.assertEqual(config.source, "当前进程环境变量")
        self.assertEqual(config.model, "test-model")

    def test_deepseek_specific_variables_receive_safe_defaults(self):
        config = resolve_model_config(
            environment={"DEEPSEEK_API_KEY": "fake-key"},
            env_files=(),
        )

        self.assertEqual(config.endpoint, "https://api.deepseek.com/chat/completions")
        self.assertEqual(config.model, "deepseek-flash")

    def test_missing_configuration_lists_checked_files(self):
        missing = Path("Z:/definitely-missing-rag-config.env")
        with self.assertRaisesRegex(RuntimeError, "definitely-missing-rag-config"):
            resolve_model_config(environment={}, env_files=(missing,))

    def test_summary_and_repr_never_include_api_key(self):
        secret = "do-not-print-this"
        config = ModelConfig(
            api_key=secret,
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash",
            source="test",
        )

        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, config_summary(config))
        self.assertIn("api.deepseek.com", config_summary(config))

    def test_env_reader_supports_comments_quotes_and_bom(self):
        with patch.object(
            Path,
            "read_text",
            return_value=(
                "\ufeff# local only\nLLM_API_KEY='quoted-key'\nLLM_MODEL_ID=\"model-id\"\n"
            ),
        ) as mocked_read:
            values = read_env_file(Path("mock.env"))

        self.assertEqual(values["LLM_API_KEY"], "quoted-key")
        self.assertEqual(values["LLM_MODEL_ID"], "model-id")
        mocked_read.assert_called_once_with(encoding="utf-8-sig")

    def test_personal_runtime_overrides_unbounded_parent_cost_settings(self):
        secret = "secret-value"
        config = ModelConfig(
            api_key=secret,
            endpoint="https://api.deepseek.com/chat/completions",
            model="deepseek-v4-flash",
            source="test",
        )
        inherited = {
            "MODEL_MAX_RETRIES": "4",
            "MODEL_RETRY_BACKOFF_SECONDS": "9",
            "MODEL_MAX_TOKENS": "8000",
        }

        with patch.dict(os.environ, inherited, clear=True):
            configure_runtime(config)
            self.assertEqual(os.environ["MODEL_MAX_RETRIES"], "0")
            self.assertEqual(os.environ["MODEL_RETRY_BACKOFF_SECONDS"], "0.5")
            self.assertEqual(os.environ["MODEL_MAX_TOKENS"], "1500")


if __name__ == "__main__":
    unittest.main()
