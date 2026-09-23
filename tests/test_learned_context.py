"""V2 acceptance, protocol isolation and paired supervision (synthetic, offline)."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from egc.io import digest, read_json, write_json, write_rows
from egc.learned import (VERSION, annotate, payload, prepare, validate,
                         validate_v1, validation_issues, visible)
from egc.learned_server import evidence_checks
from test_learned import fixture, annotation
from test_learned_retry import response


class ContextTests(unittest.TestCase):
    def test_cross_sentence_alias_long_and_repeated_quotes_are_not_rejected(self):
        quote = "随后" + "加工材料"*41 + "。"
        row = {"facts": "甲租用了仓库。"+quote+quote}
        body = {"decision": "keep", "issues": [], "summary": "摘要"*110,
                "evidence": [{"quote": quote, "subject": "被告人甲", "relation": "target", "kind": "action"}]}
        before = copy.deepcopy(body)
        self.assertIs(validate(body, row), body)
        self.assertEqual(body, before)
        with self.assertRaises(ValueError):
            validate_v1(body, row)
        body["evidence"][0]["subject"] = None
        self.assertEqual(validation_issues(body, row), [])
        # Acceptance makes no assertion that the attribution is semantically true.

    def test_missing_fields_and_rewritten_quotes_still_require_model_correction(self):
        row = {"facts": "甲租用仓库。随后加工材料。"}
        body = {"decision": "keep", "issues": [], "summary": "摘要", "evidence": [
            {"quote": "甲随后加工材料。", "relation": "target", "kind": "action"}]}
        before = copy.deepcopy(body)
        errors = validation_issues(body, row)
        self.assertEqual(errors[0]["missing"], ["subject"])
        self.assertEqual(errors[1]["problem"], "quote_not_in_source")
        with self.assertRaises(ValueError):
            validate(body, row)
        self.assertEqual(body, before)

    def test_v1_directory_is_refused_before_api_or_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            old = root / "old"
            identity = {"version": "learned-evidence-v1", "inputs_hash": digest(rows)}
            write_json(old / "identity.json", identity)
            with patch("egc.learned.call_deepseek") as api, patch("egc.credentials.read_key") as key:
                with self.assertRaisesRegex(ValueError, "new annotation directory"):
                    annotate(pool, old)
                api.assert_not_called()
                key.assert_not_called()
            self.assertEqual(read_json(old / "identity.json"), identity)

    def test_context_reply_survives_packaging_with_shared_source_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            def api(request, key, **kwargs):
                r = json.loads(request["messages"][1]["content"])
                body = annotation(r)
                body["evidence"][0]["quote"] = r["facts"].split("。", 1)[1]
                body["evidence"][0]["subject"] = "被告人甲"
                return response(body, request, kwargs)
            with patch("egc.learned.call_deepseek", side_effect=api) as calls, patch("egc.credentials.read_key", return_value="TEST_ONLY"):
                result = annotate(pool, root / "annotations", workers=1)
            self.assertEqual(result["status"], {"keep": 2})
            self.assertEqual(calls.call_count, 2)
            result = prepare(pool, root / "annotations", root / "ready")
            self.assertEqual(result["version"], VERSION)
            with zipfile.ZipFile(root / "ready/experiment.zip") as z:
                bound = json.loads(z.read("bound.train.sft.jsonl"))
                flat = json.loads(z.read("flat.train.sft.jsonl"))
                b = json.loads(bound["completion"][0]["content"])
                f = json.loads(flat["completion"][0]["content"])
                self.assertEqual(f["evidence"], [e["quote"] for e in b["evidence"]])
                self.assertEqual(b["evidence"][0]["subject"], "被告人甲")
                self.assertEqual(b["sentence_months"], rows[0]["sentence_months"])
                self.assertIn("不必在quote中出现", bound["prompt"][0]["content"])
            pred = {"id": rows[0]["id"], "text": json.dumps(b), "finish_reason": "stop"}
            diagnostics = evidence_checks(rows, [pred], "bound")
            self.assertEqual(diagnostics["source_quotes"], 1)
            self.assertIsNone(diagnostics["semantic_attribution_accuracy"])
