import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egc import facts
from egc.fact_experiment import prepare, run, report_run, summarize
from egc.io import read_json, read_rows, write_json, write_rows
from egc.annotate import call_deepseek
from test_pipeline import row


PROFILE = {"version": "test-v1", "conditions": [{"id": "returned_or_compensated", "text": "被告已经退赔"}]}


def case(number=1, text="被告人甲的家属代甲退还款项。", charge="诈骗罪"):
    return {**row(number=number, facts=text), "charge": charge}


def observation(oid="o1", relation="proxy", polarity="supports"):
    quote = "被告人甲的家属代甲退还款项。"
    return {"id": oid, "condition_id": "returned_or_compensated",
        "subject": {"relation": relation, "quote": quote},
        "event": {"scope": "current_case", "coverage": "single_event", "quote": quote},
        "time_quote": None, "source": {"kind": "unspecified", "quote": None},
        "polarity": polarity, "quotes": [quote]}


def body(obs=None, reason="none"):
    obs = obs if obs is not None else [observation()]
    return {"observations": obs, "resolutions": [{"condition_id": "returned_or_compensated",
                                                   "observation_ids": [o["id"] for o in obs], "reason": reason}]}


def envelope(content=None, finish="stop"):
    return {"id": "test-response", "model": "test-model", "created": 0,
            "choices": [{"finish_reason": finish, "message": {"content": json.dumps(content or body())}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}}


def fake_call(payload, key, metadata, transport, capture_response):
    capture_response(envelope())
    return body(), {}, {}


class FactSchemaTests(unittest.TestCase):
    def test_model_payload_does_not_contain_reference_or_labels_or_expectations(self):
        source = case()
        altered = {**source, "sentence_months": 10000, "opinion": "secret", "expected": {"x": "supported"}}
        for mode in facts.MODES:
            self.assertEqual(facts.payload(source, PROFILE, mode, "test", 100),
                             facts.payload(altered, PROFILE, mode, "test", 100))

    def test_proxy_compensation_supported_but_other_actor_is_unknown(self):
        self.assertEqual(facts.validate(body(), case(), PROFILE, "bound")["annotation"]["conditions"][0]["status"], "supported")
        output = facts.validate(body([observation(relation="other")]), case(), PROFILE, "bound")
        self.assertEqual(output["annotation"]["conditions"][0]["status"], "unknown")
        self.assertEqual(output["derivation"][0]["reason"], "binding_gate")

    def test_conflicting_evidence_not_resolved_by_order(self):
        data = body([observation(), observation("o2", polarity="refutes")])
        output = facts.validate(data, case(), PROFILE, "bound")
        self.assertEqual(output["annotation"]["conditions"][0]["status"], "unknown")
        data["observations"].reverse()
        self.assertEqual(facts.validate(data, case(), PROFILE, "bound")["annotation"], output["annotation"])

    def test_missing_binding_quote_and_false_time_fail(self):
        for field in ("subject", "source"):
            data = body()
            if field == "source":
                data["observations"][0][field]["kind"] = "court_finding"
            data["observations"][0][field]["quote"] = None
            with self.assertRaises(ValueError):
                facts.validate(data, case(), PROFILE, "bound")
        data = body()
        data["observations"][0]["time_quote"] = "2020年"
        with self.assertRaises(ValueError):
            facts.validate(data, case(), PROFILE, "bound")

    def test_unreferenced_or_wrong_condition_observation_rejected(self):
        data = body()
        data["resolutions"][0].update(observation_ids=[], reason="not_mentioned")
        with self.assertRaisesRegex(ValueError, "Orphan"):
            facts.validate(data, case(), PROFILE, "bound")

    def test_single_event_cannot_establish_case_wide_noncompletion(self):
        profile = {"conditions": [{"id": "unsuccessful_completion", "text": "全案未得逞"}]}
        data = body([observation(relation="defendant")])
        data["observations"][0]["condition_id"] = "unsuccessful_completion"
        data["resolutions"][0]["condition_id"] = "unsuccessful_completion"
        self.assertEqual(facts.validate(data, case(), profile, "bound")["annotation"]["conditions"][0]["status"], "unknown")
        data["observations"][0]["event"]["coverage"] = "case_wide"
        self.assertEqual(facts.validate(data, case(), profile, "bound")["annotation"]["conditions"][0]["status"], "supported")


class FactRunTests(unittest.TestCase):
    def test_cli_three_mode_matrix_and_report_with_mocked_api(self):
        from egc.cli import parser, run as cli_run
        def respond(request, key, metadata, transport, capture_response):
            conditions = [{"condition_id": "returned_or_compensated", "status": "supported",
                           "quotes": ["被告人甲的家属代甲退还款项。"]}]
            prompt = request["messages"][0]["content"]
            content = body() if prompt == facts.BOUND else {"conditions": conditions}
            if prompt not in {facts.BOUND, facts.FLAT}:
                content["claims"] = [{"text": "明确记载家属代退赔。", "quotes": conditions[0]["quotes"]}]
            capture_response(envelope(content))
            return content, {}, {}
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture"}), \
                patch("egc.fact_experiment.call_deepseek", side_effect=respond) as call:
            tmp = Path(temp)
            write_rows(tmp / "input.jsonl", [case()])
            write_json(tmp / "profile.json", PROFILE)
            write_json(tmp / "expected.json", {"items": [{"id": case()["id"],
                "expected": {"returned_or_compensated": "supported"}}]})
            result = cli_run(parser().parse_args(["facts-run", "--input", str(tmp / "input.jsonl"),
                "--profile", str(tmp / "profile.json"), "--output-dir", str(tmp / "run")]))
            self.assertEqual(result["statuses"], {"accepted": 3})
            self.assertEqual(call.call_count, 3)
            result = cli_run(parser().parse_args(["facts-report", "--input", str(tmp / "input.jsonl"),
                "--run-dir", str(tmp / "run"), "--expectations", str(tmp / "expected.json"),
                "--output", str(tmp / "comparison.json")]))
            self.assertEqual(result["regression_agreement_not_accuracy"]["bound"]["matched"], 1)
            self.assertIsNone(result["semantic_accuracy"])

    def test_dry_run_has_no_calls_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp, patch("egc.fact_experiment.call_deepseek") as call:
            out = Path(tmp) / "new"
            result = run([case()], PROFILE, out, limit=2, dry_run=True)
            self.assertEqual(result["new_requests_this_run_at_most"], 2)
            self.assertFalse(out.exists())
            call.assert_not_called()

    def test_bounded_cache_resume_and_original_labels_protected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture"}), \
                patch("egc.fact_experiment.call_deepseek", side_effect=fake_call) as call:
            rows = [case(), case(number=2)]
            run(rows, PROFILE, tmp, modes=["bound"], limit=1)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(read_json(Path(tmp) / "manifest.json")["statuses"], {"accepted": 1, "pending": 1})
            run(rows, PROFILE, tmp, modes=["bound"], limit=1)
            run(rows, PROFILE, tmp, modes=["bound"], limit=1)
            self.assertEqual(call.call_count, 2)
            changed = copy.deepcopy(rows)
            changed[0]["sentence_months"] += 1
            with self.assertRaisesRegex(ValueError, "identity"):
                run(changed, PROFILE, tmp, modes=["bound"], limit=1)

    def test_invalid_response_cached_and_agreement_counts_failure(self):
        def bad(*args, capture_response, **kwargs):
            capture_response(envelope(finish="length"))
            raise ValueError("Incomplete")
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture"}), \
                patch("egc.fact_experiment.call_deepseek", side_effect=bad) as call:
            source = case()
            run([source], PROFILE, tmp, modes=["bound"])
            run([source], PROFILE, tmp, modes=["bound"])
            self.assertEqual(call.call_count, 1)
            result = report_run([source], tmp, Path(tmp) / "comparison.json",
                {"items": [{"id": source["id"], "expected": {"returned_or_compensated": "supported"}}]})
            self.assertEqual(result["regression_agreement_not_accuracy"]["bound"]["total"], 1)
            self.assertEqual(result["regression_agreement_not_accuracy"]["bound"]["matched"], 0)

    def test_network_failure_stops_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture"}), \
                patch("egc.fact_experiment.call_deepseek", side_effect=RuntimeError("network")) as call:
            rows = [case()]
            run(rows, PROFILE, tmp, modes=["bound"])
            run(rows, PROFILE, tmp, modes=["bound"])
            self.assertEqual(call.call_count, 1)
            self.assertEqual(read_rows(Path(tmp) / "records.jsonl")[0]["status"], "api_error")

    def test_test_data_refused_and_expectation_not_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "train only"):
                run([{**case(), "split": "test"}], PROFILE, tmp, dry_run=True)

    def test_capture_happens_before_content_validation(self):
        from types import SimpleNamespace
        response = envelope(finish="length")
        captured = []
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(response) + "\n200")):
            with self.assertRaises(ValueError):
                call_deepseek({}, "fixture", transport="curl", capture_response=captured.append)
        self.assertEqual(captured, [response])

    def test_prepare_excludes_exposed_source_and_keeps_original_split(self):
        known = case(1)
        fresh = case(2, "这是独立未暴露的诈骗合成样例，内容不与旧案重复。")
        robbery = case(3, "独立抢劫合成案件用于验证筛选，不涉及任何真实人物。", "抢劫罪")
        regression = {"items": [{"id": known["id"], "expected": {"returned_or_compensated": "supported"}}]}
        with tempfile.TemporaryDirectory() as tmp:
            prepare([known, fresh, robbery], [known], regression, PROFILE, tmp, per_charge=1)
            self.assertEqual({r["id"] for r in read_rows(Path(tmp) / "new.jsonl")}, {fresh["id"], robbery["id"]})
            self.assertEqual(read_rows(Path(tmp) / "regression.jsonl"), [known])


if __name__ == "__main__":
    unittest.main()
