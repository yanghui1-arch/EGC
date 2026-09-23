import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from egc.annotate import ApiRequestError
from egc.credentials import read_key
from egc.io import digest, read_json, write_json, write_rows
from egc.learned import annotate, payload, visible, validation_issues_v1
from test_learned import fixture, annotation


def response(body, request, kwargs):
    envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body, ensure_ascii=False)}}],
                "model": "test-model", "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
    kwargs["capture_response"](envelope)
    return body, envelope["usage"], {"model": "test-model"}


class RetryTests(unittest.TestCase):
    def test_detailed_feedback_reports_all_fields_without_changing_output(self):
        quoted = "甲" + "证"*160 + "。"
        r = {"facts": quoted + "另一个片段。"}
        body = {"decision": "keep", "issues": [], "summary": "摘要", "evidence": [
            {"quote": quoted, "subject": "乙", "relation": "target", "kind": "action"},
            {"quote": "另一个片段。", "subject": None, "relation": "target", "kind": "outcome"}]}
        before = json.dumps(body)
        issues = validation_issues_v1(body, r)
        self.assertEqual(json.dumps(body), before)
        self.assertEqual(len(issues), 3)
        self.assertEqual(issues[0], {"path": "evidence[0].quote", "problem": "too_long", "actual_chars": 162, "maximum_chars": 160})
        self.assertEqual(issues[1]["path"], "evidence[0].subject")
        self.assertEqual(issues[2]["path"], "evidence[1].subject")

    def test_two_rejected_model_outputs_do_not_stop_single_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            extra = dict(rows[0], id="extra", facts="甲独立样本。", source_id="extra")
            write_rows(pool / "train.jsonl", [rows[0], extra])
            manifest = read_json(pool / "manifest.json")
            manifest["train_hash"] = digest([rows[0], extra])
            write_json(pool / "manifest.json", manifest)
            def api(request, key, **kwargs):
                return response({"decision": "keep"}, request, kwargs)
            with patch("egc.credentials.read_key", return_value="TEST_ONLY"), patch("egc.learned.call_deepseek", side_effect=api) as call:
                report = annotate(pool, root / "cache", max_attempts=1, workers=1, limit=10)
            self.assertEqual(call.call_count, 3)
            self.assertFalse(report["stopped_on_repeated_failures"])
            self.assertEqual(report["this_run_failure_stages"], {"annotation_validation": 3})
            self.assertEqual(report["stop_reason"], "eligible_cases_finished")

    def test_missing_fields_are_regenerated_not_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            calls = []
            def api(request, key, **kwargs):
                calls.append(request)
                if len(calls) == 1:
                    body = {"decision": "keep"}
                else:
                    body = annotation(rows[0])
                return response(body, request, kwargs)
            with patch("egc.credentials.read_key", return_value="TEST_ONLY"), patch("egc.learned.call_deepseek", side_effect=api):
                report = annotate(pool, root / "cache", limit=2, workers=1)
            self.assertEqual(report["new_requests"], 2)
            self.assertEqual(report["cases_processed_this_run"], 1)
            self.assertEqual(len(calls[1]["messages"]), 4)
            self.assertIn("annotation_schema", calls[1]["messages"][-1]["content"])
            base = digest(payload(rows[0], "deepseek-flash"))
            history = root / "cache/attempts" / base
            first, second = read_json(history / "001.json"), read_json(history / "002.json")
            self.assertEqual(first["status"], "failed")
            self.assertNotIn("body", first)
            self.assertEqual(second["body"], annotation(rows[0]))
            self.assertEqual(second["attempts_used"], 2)
            self.assertNotIn("TEST_ONLY", (history / "002.json").read_text(encoding="utf-8"))

    def test_retry_cap_global_budget_and_resume_no_rebilling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            def api(request, key, **kwargs):
                return response({"decision": "keep"}, request, kwargs)
            with patch("egc.credentials.read_key", return_value="TEST_ONLY"), patch("egc.learned.call_deepseek", side_effect=api) as call:
                a = annotate(pool, root / "cache", limit=4, workers=2)
                self.assertEqual(call.call_count, 4)
                self.assertEqual(a["new_requests"], 4)
                b = annotate(pool, root / "cache", limit=100, workers=2, retry_failed=True, failed_only=True)
                self.assertEqual(call.call_count, 6)
                self.assertEqual(b["status"]["failed"], 2)
                c = annotate(pool, root / "cache", retry_failed=True, failed_only=True)
                self.assertEqual(call.call_count, 6)
                self.assertEqual(c["capped_failures_not_retried"], 2)

    def test_legacy_failed_response_retained_and_successes_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            base = digest(payload(rows[0], "deepseek-flash"))
            legacy = {"id": rows[0]["id"], "status": "failed", "error_type": "ValueError",
                      "input_hash": digest(visible(rows[0])), "request_hash": base,
                      "response": {"choices": [{"message": {"content": '{"decision":"keep"}'}}]}}
            write_json(root / "cache/cache" / (base+".json"), legacy)
            def api(request, key, **kwargs):
                return response(annotation(rows[0]), request, kwargs)
            with patch("egc.credentials.read_key", return_value="TEST_ONLY"), patch("egc.learned.call_deepseek", side_effect=api) as call:
                report = annotate(pool, root / "cache", retry_failed=True, failed_only=True)
                annotate(pool, root / "cache", retry_failed=True, failed_only=True)
                self.assertEqual(call.call_count, 1)
            self.assertEqual(report["status"]["keep"], 1)
            self.assertEqual(report["status"]["failed"], 0)
            self.assertEqual(read_json(root / "cache/attempts" / base / "000.json"), legacy)
            self.assertFalse((root / "cache/cache" / (digest(payload(rows[1], "deepseek-flash"))+".json")).exists())

    def test_auth_failure_stops_and_never_loops_on_bad_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, _ = fixture(root)
            with patch("egc.credentials.read_key", return_value="TEST_ONLY"), patch("egc.learned.call_deepseek", side_effect=ApiRequestError("http_status", 401)) as call:
                report = annotate(pool, root / "cache", workers=1)
            self.assertTrue(report["stopped_on_auth_error"])
            self.assertEqual(call.call_count, 1)

    def test_invalid_subject_is_not_replaced_with_null_or_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            bad = annotation(rows[0])
            bad["evidence"][0]["subject"] = 12
            def api(request, key, **kwargs):
                return response(bad, request, kwargs)
            with patch("egc.credentials.read_key", return_value="TEST_ONLY"), patch("egc.learned.call_deepseek", side_effect=api):
                annotate(pool, root / "cache", workers=1, limit=3)
            base = digest(payload(rows[0], "deepseek-flash"))
            saved = read_json(root / "cache/cache" / (base+".json"))
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["attempts_used"], 3)
            self.assertNotIn("body", saved)
            preserved = json.loads(saved["response"]["choices"][0]["message"]["content"])
            self.assertEqual(preserved["evidence"][0]["subject"], 12)
            self.assertEqual(preserved["evidence"][0]["relation"], "target")

    def test_key_shown_only_to_direct_console_not_logging_streams(self):
        class Console(io.StringIO):
            def close(self):
                pass
        terminal, logged = Console(), io.StringIO()
        with redirect_stdout(logged), redirect_stderr(logged), patch("egc.credentials.getpass.getpass", return_value="TEST_KEY_ONLY"), \
             patch("egc.credentials.sys.stdin.isatty", return_value=True), patch.object(logged, "isatty", return_value=True), \
             patch("builtins.open", return_value=terminal), patch("builtins.input", return_value="y"):
            self.assertEqual(read_key(True, True), "TEST_KEY_ONLY")
        self.assertIn("TEST_KEY_ONLY", terminal.getvalue())
        self.assertNotIn("TEST_KEY_ONLY", logged.getvalue())
        with patch("egc.credentials.getpass.getpass", return_value="BAD KEY"):
            with self.assertRaisesRegex(ValueError, "whitespace"):
                read_key(True)
        with patch("egc.credentials.getpass.getpass", return_value="TEST_KEY_ONLY"), patch("egc.credentials.sys.stdin.isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "interactive"):
                read_key(True, True)


if __name__ == "__main__":
    unittest.main()
