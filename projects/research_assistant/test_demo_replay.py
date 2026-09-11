"""Replay runs real local code, exports only selected excerpts, and works offline."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

from demo_replay import (DEMO, backend_case, build_payload, check_public_text, job_cases,
                         load_sources, render_html, validate_url, without_network)


class ReplayTests(unittest.TestCase):
    def test_project_local_agent_import_works_in_fresh_process(self):
        code = """import agent_loop, assistant, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1] / 'experiments/02-openai-compatible-adapter'))
import openai_compatible
assert assistant.AgentResult is agent_loop.AgentResult
assert openai_compatible.ModelResponse is agent_loop.ModelResponse
assert agent_loop.ToolCall('example', {}).name == 'example'
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=DEMO.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_retrieval_and_empty_scope_keep_source_contract(self):
        source = load_sources()
        answer, empty = job_cases(source)
        self.assertEqual(len(answer["evidence"]), 2)
        self.assertFalse(answer["raw_result"]["refused"])
        for evidence in answer["evidence"]:
            self.assertIn(evidence["text"], answer["result"])
        self.assertTrue(empty["raw_result"]["refused"])
        self.assertEqual(empty["raw_result"]["failure_type"], "retrieval_empty")
        self.assertEqual(empty["evidence"], [])

    def test_backend_replay_persists_and_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as directory, without_network():
            case = backend_case(load_sources(), Path(directory))
            self.assertEqual(case["checks"]["generator_calls"], 1)
            self.assertTrue(case["checks"]["reopened_request_replayed"])
            self.assertTrue(case["checks"]["stale_review_conflict"])
            self.assertEqual(len(list(Path(directory).glob("*.sqlite"))), 1)
            self.assertNotIn(directory, json.dumps(case))

    def test_export_rejects_contacts_keys_and_local_paths(self):
        examples = ["mail:" + "test" + "@" + "example.org", "phone:" + "138" + "00138000",
                    "sk-" + "x" * 30, "C:" + "\\Users\\someone\\private.csv"]
        for value in examples:
            with self.subTest(value=value[:6]), self.assertRaises(ValueError):
                check_public_text(value)
        check_public_text("https://documentation.espressif.com/a.pdf")
        check_public_text("a12ff" + "138" + "00138000" + "ac77")

    def test_unknown_fields_cannot_silently_enter_selected_sources(self):
        source = copy.deepcopy(load_sources())
        source["jobs"][0]["private_notes"] = "not for export"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected job"):
                load_sources(path)

    def test_source_links_reject_credentials_unapproved_hosts_and_schemes(self):
        for url in ["javascript:alert(1)", "https://test" + "@" + "example.org/x", "http://talent.baidu.com/x",
                    "https://talent.baidu.com/x?token=example", "https://example.org/data"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url)

    def test_html_payload_cannot_close_script_element(self):
        payload = {"text": "</script><script>alert('example')</script>\u2028"}
        html = render_html(payload)
        encoded = html.split('<script type="application/json" id="demo-data">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(encoded), payload)
        self.assertNotIn("<script>alert", html)

    def test_network_connections_are_denied_during_generation(self):
        with without_network():
            with self.assertRaisesRegex(RuntimeError, "network"):
                socket.create_connection(("example.org", 443))

    def test_built_page_embeds_data_and_has_no_external_assets(self):
        class Assets(HTMLParser):
            def __init__(self):
                super().__init__()
                self.external = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag in {"script", "img", "iframe", "link", "audio", "video"}:
                    self.external.extend(v for k, v in attrs.items() if k in {"src", "href"})
        with tempfile.TemporaryDirectory() as directory:
            payload = build_payload(load_sources(), Path(directory))
        html = render_html(payload)
        parser = Assets()
        parser.feed(html)
        self.assertEqual(parser.external, [])
        self.assertEqual(len(payload["cases"]), 4)
        self.assertEqual(payload["new_model_calls"], 0)
        self.assertEqual(payload["cases"][-1]["recording"]["recorded_at"], "2026-09-09")
        self.assertIn("connect-src 'none'", html)


if __name__ == "__main__":
    unittest.main()
