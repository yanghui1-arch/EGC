import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from egc.annotate import annotate, request_payload
from egc.data import audit, canonicalize, overlap_report, pilot_review, split_train, visible_case
from egc.evaluate import compare, parse_output, summarize
from egc.io import digest, read_json, read_rows, write_rows
from egc.prepare import prepare
from egc.rules import (annotation_fingerprint, compose, evaluate, import_chains, resolve_quotes,
                       select_rules, validate_annotation, validate_rules)
from egc.server import check_training_rows, resolve_model

ROOT = Path(__file__).resolve().parents[1]


def row(split="train", number=0, facts="当事人实施了合成行为A。已返还合成物品。"):
    return canonicalize([{"filename": str(number), "caseCause": "合成测试罪名", "justice": facts,
                          "opinion": "参考裁判秘密不能进入输入", "judge": 73}], "synthetic", split)[0]


def bundle():
    return read_json(ROOT / "examples/synthetic_rules.json")


def annotation(case, rules=None):
    selected = select_rules(rules or bundle(), case["charge"])
    return resolve_quotes({"conditions": [
        {"condition_id": "action", "status": "supported", "quotes": ["实施了合成行为A"]},
        {"condition_id": "returned", "status": "unknown", "quotes": []}]}, case, selected)


class DataTests(unittest.TestCase):
    def test_only_explicit_months(self):
        for bad in (True, -1, 1.5, "三年", float("nan")):
            with self.assertRaises(ValueError):
                canonicalize([dict(justice="事实", caseCause="罪名", judge=bad)], "x", "train")

    def test_cail_unknown_reference_is_missing(self):
        value = canonicalize([dict(justice="事实", caseCause="罪名", judge=5, opinion="未知")], "cail", "test")[0]
        self.assertIsNone(value["opinion"])

    def test_facts_preserved_for_character_offsets(self):
        text = "  第一行\n第二行。"
        self.assertEqual(row(facts=text)["facts"], text)

    def test_test_set_never_becomes_train_or_pilot(self):
        for function in (split_train, pilot_review):
            with self.assertRaises(ValueError):
                function([row("test")])

    def test_grouped_split_and_determinism(self):
        rows = [row(number=i, facts=f"案情{i}。") for i in range(20)]
        rows.append(dict(rows[0], id="duplicate-facts", source_id="alias"))
        rows.append(dict(rows[1], id="duplicate-source", facts="该案另一个文本版本"))
        train, dev = split_train(rows, .25, 42)
        self.assertEqual((train, dev), split_train(list(reversed(rows)), .25, 42))
        self.assertFalse(any(overlap_report({"train": train, "dev": dev}).values()))
        self.assertEqual(len(train)+len(dev), len(rows))

    def test_duplicate_case_ids_fail(self):
        with self.assertRaises(ValueError):
            audit([row(), row()])

    def test_flags_are_not_gold_sufficiency_labels(self):
        case = row(facts="查明事实。")
        case["opinion"] = "被告自首。"
        report = audit([case])
        self.assertEqual(report["review_flags"][0]["opinion_only_keywords"], ["自首"])
        self.assertIn("not_a_gold", report["review_flags"][0]["interpretation"])


class RuleTests(unittest.TestCase):
    def test_three_valued_logic(self):
        states = {"yes": "supported", "no": "refuted", "u": "unknown"}
        cases = [({"all": ["yes", "u"]}, "unknown"), ({"all": ["no", "u"]}, "refuted"),
                 ({"any": ["yes", "u"]}, "supported"), ({"any": ["no", "u"]}, "unknown"),
                 ({"not": "u"}, "unknown"), ({"not": "no"}, "supported")]
        for expression, expected in cases:
            self.assertEqual(evaluate(expression, states), expected)

    def test_missing_condition_does_not_activate_modifier(self):
        trace = compose(bundle(), annotation(row()))
        self.assertEqual([r["status"] for r in trace["rules"]], ["supported", "unknown"])

    def test_unknown_defeater_prevents_asserted_conclusion(self):
        rules = bundle()
        rules["rules"][1]["requires"] = []
        rules["rules"][0]["blocked_by"] = ["synthetic_modifier"]
        validate_rules(rules)
        self.assertEqual(compose(rules, annotation(row(), rules))["rules"][0]["status"], "unknown")

    def test_cycle_and_unknown_condition_rejected(self):
        rules = bundle()
        rules["rules"][0]["requires"] = ["synthetic_modifier"]
        with self.assertRaises(ValueError):
            validate_rules(rules)
        rules = bundle()
        rules["rules"][0]["when"] = "not-in-schema"
        with self.assertRaises(ValueError):
            validate_rules(rules)

    def test_draft_rules_fail_closed(self):
        rules = bundle()
        rules["rules"][0]["review_status"] = "draft"
        with self.assertRaises(ValueError):
            validate_rules(rules)
        validate_rules(rules, allow_draft=True)

    def test_import_preserves_compound_condition(self):
        rules = import_chains("行为 -> 情形A OR 情形B -> 后果", "测试", "source", "v1")
        self.assertEqual(rules["conditions"][1]["text"], "情形A OR 情形B")
        self.assertEqual(rules["rules"][0]["review_status"], "draft")

    def test_invalid_and_ambiguous_quotes_rejected(self):
        for facts, quote in [("输入原文", "编造事实"), ("原文原文", "原文")]:
            with self.assertRaises(ValueError):
                resolve_quotes({"conditions": [{"condition_id": "action", "status": "supported", "quotes": [quote]}]}, row(facts=facts), bundle())

    def test_annotation_cannot_follow_modified_input(self):
        original = row()
        record = annotation(original)
        changed = dict(original, facts=original["facts"] + "其他内容")
        with self.assertRaises(ValueError):
            validate_annotation(record, changed, bundle())

    def test_unsupported_annotation_needs_evidence(self):
        record = annotation(row())
        record["conditions"][0]["evidence"] = []
        with self.assertRaises(ValueError):
            validate_annotation(record, row(), bundle())


class LeakageTests(unittest.TestCase):
    def test_all_variants_ignore_gold_answer(self):
        case = row()
        record = annotation(case)
        for variant in ("base", "rules", "concat", "egc"):
            jobs, sft, _ = prepare([case], variant, bundle(), [record])
            changed = dict(case, opinion="完全不同的标签", sentence_months=999)
            changed_jobs, _, _ = prepare([changed], variant, bundle(), [record])
            self.assertEqual(jobs, changed_jobs)
            self.assertNotIn("参考裁判秘密", json.dumps(jobs, ensure_ascii=False))
            self.assertIn("参考裁判秘密", sft[0]["completion"][0]["content"])

    def test_deepseek_payload_and_cache_fingerprint_ignore_gold(self):
        case, rules = row(), bundle()
        changed = dict(case, opinion="不同参考", sentence_months=456)
        self.assertEqual(request_payload(case, rules, "test-model", 100), request_payload(changed, rules, "test-model", 100))
        self.assertEqual(annotation_fingerprint(case, rules), annotation_fingerprint(changed, rules))

    def test_test_preparation_has_no_sft_targets(self):
        jobs, sft, _ = prepare([row("test")], "base")
        self.assertEqual(sft, [])
        self.assertNotIn("completion", jobs[0])

    def test_sft_rejects_test_validation(self):
        _, sft, _ = prepare([row()], "base")
        with self.assertRaises(ValueError):
            check_training_rows(sft, [dict(sft[0], split="test")])

    def test_sft_rejects_same_source_under_changed_text(self):
        _, train, _ = prepare([row(number=7)], "base")
        _, dev, _ = prepare([row("dev", number=7, facts="该案件的不同写法")], "base")
        dev[0]["id"] = "distinct-id-same-source"
        with self.assertRaisesRegex(ValueError, "same-source"):
            check_training_rows(train, dev)

    def test_api_cache_and_request_limit(self):
        rows = [row(number=i, facts=f"记录{i}。当事人实施了合成行为A。") for i in range(3)]
        payload = {"conditions": [{"condition_id": "action", "status": "supported", "quotes": ["实施了合成行为A"]},
                                  {"condition_id": "returned", "status": "unknown", "quotes": []}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake-test-credential"}), patch("egc.annotate.call_deepseek", return_value=(payload, {})) as call:
            output, cache = Path(tmp)/"ann.jsonl", Path(tmp)/"cache"
            report = annotate(rows, bundle(), output, cache, "test-model", limit=1)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(report["saved_cases"], 1)
            annotate(rows, bundle(), output, cache, "test-model", limit=1)
            self.assertEqual(call.call_count, 2)
            with self.assertRaises(ValueError):
                annotate(rows, bundle(), output, cache, "another-model")
            dry = annotate(rows, bundle(), output, cache, "test-model", dry_run=True)
            self.assertEqual(dry["new_cases_planned"], 1)
            self.assertEqual(call.call_count, 2)


class EvaluationTests(unittest.TestCase):
    def test_no_numeric_guess_or_zero_fallback(self):
        for text in ('{"reasoning":"文本","sentence_months":true}', '{"reasoning":"文本","sentence_months":"36"}',
                     "可能是24或36个月", '{"sentence_months":36}', '{"reasoning":"文本","sentence_months":-1}'):
            self.assertIsNone(parse_output(text))

    def test_partial_coverage_not_full_mae(self):
        rows = [row(number=i, facts=f"事实{i}") for i in range(2)]
        pred = [{"id": rows[0]["id"], "text": '{"reasoning":"文本","sentence_months":73}'}]
        report = summarize(rows, pred)
        self.assertEqual(report["coverage"], .5)
        self.assertIsNone(report["mae_months_full"])
        self.assertEqual(report["mae_months_valid_only"], 0)
        with self.assertRaises(ValueError):
            compare(rows, pred, pred)

    def test_paired_bootstrap_direction(self):
        rows = [row(number=i, facts=f"事实{i}") for i in range(3)]
        def outputs(months):
            return [{"id": r["id"], "text": json.dumps({"reasoning": "文本", "sentence_months": months})} for r in rows]
        report = compare(rows, outputs(83), outputs(78), samples=100)
        self.assertEqual(report["delta_mae_candidate_minus_baseline"], -5)
        self.assertEqual(report["paired_case_bootstrap_ci95"], [-5, -5])

    def test_unknown_ids_rejected(self):
        with self.assertRaises(ValueError):
            summarize([row()], [{"id": "foreign", "text": ""}])

    def test_model_directory_must_be_explicit(self):
        with self.assertRaises(ValueError):
            resolve_model("a-hub-model-name-not-a-local-directory")


if __name__ == "__main__":
    unittest.main()
