from __future__ import annotations

import unittest

from run_live_web_scenarios import (
    DEFAULT_SCENARIOS,
    Scenario,
    validate_http_result,
    validate_scenario,
)


class LiveWebScenarioTests(unittest.TestCase):
    def test_grounded_real_model_response_passes_contract(self):
        scenario = Scenario("case", "问题", top_k=3)
        payload = {
            "question": "问题",
            "answer_mode": "rag",
            "answer": "证据结论 [1]。",
            "refused": False,
            "failure_type": None,
            "citations": [{"number": 1}],
            "model_usage": {"total_tokens": 42},
        }

        self.assertEqual(validate_scenario(scenario, payload), [])

    def test_grounded_contract_detects_missing_usage_and_marker(self):
        scenario = Scenario("case", "问题", top_k=3)
        payload = {
            "question": "问题",
            "answer_mode": "rag",
            "answer": "没有编号的回答",
            "refused": False,
            "citations": [{"number": 2}],
            "model_usage": None,
        }

        failures = validate_scenario(scenario, payload)

        self.assertIn("answer is missing citation marker [2]", failures)
        self.assertIn("real-model answer has no provider usage", failures)

    def test_market_analysis_must_not_report_model_usage(self):
        scenario = Scenario(
            "market",
            "统计",
            top_k=3,
            expected_mode="market_analysis",
            require_model_usage=False,
        )
        payload = {
            "question": "统计",
            "answer_mode": "market_analysis",
            "answer": "统计结果",
            "refused": False,
            "citations": [],
            "model_usage": {"total_tokens": 1},
        }

        self.assertIn(
            "deterministic market analysis unexpectedly used the model",
            validate_scenario(scenario, payload),
        )

    def test_default_suite_caps_paid_rag_cases_at_four(self):
        self.assertEqual(len(DEFAULT_SCENARIOS), 5)
        self.assertEqual(
            sum(1 for item in DEFAULT_SCENARIOS if item.expected_mode == "rag"),
            4,
        )

    def test_explicit_output_truncation_is_recorded_as_safe_failure(self):
        scenario = Scenario(
            "comparison",
            "复杂比较",
            top_k=5,
            allowed_failure_types=("output_truncated",),
        )
        payload = {
            "detail": {
                "failure_type": "output_truncated",
                "model_usage": {"total_tokens": 2500},
            }
        }

        failures, outcome = validate_http_result(scenario, 503, payload)

        self.assertEqual(failures, [])
        self.assertEqual(outcome, "safe_failure")


if __name__ == "__main__":
    unittest.main()
