import copy
import unittest
import tempfile
from pathlib import Path

from egc.annotation_audit import audit, risk_flags
from egc.cail import input_risks
from egc.distill import validate
from test_cail_distill import BODY, PROFILE
from test_pipeline import row
from egc.io import read_rows, write_rows
from scripts.screen_pool import screen


def pair():
    original = row()
    original["facts"] = "被告人实施了合成行为A，其他事实详见此独立材料。"
    original["opinion"] = None
    result = validate(BODY, original, PROFILE)
    annotated = {**original, "opinion": result["synthetic_opinion"], "distillation": result,
        "opinion_provenance": {"kind": "synthetic_source_only", "reference_labels_sent_to_teacher": False}}
    return original, annotated


class AnnotationAuditTests(unittest.TestCase):
    def test_screen_preserves_rows_and_splits_and_quarantines_intact(self):
        source, _ = pair()
        risky = {**source, "id": "appeal_case", "facts": "经二审查明，上诉人已赔偿。"}
        with tempfile.TemporaryDirectory() as temp:
            inp, out = Path(temp) / "source.jsonl", Path(temp) / "screened"
            write_rows(inp, [source, risky])
            report = screen([inp], out)
            self.assertEqual(read_rows(out / (source["split"] + ".jsonl")), [source])
            self.assertEqual(read_rows(out / "quarantine.jsonl")[0]["case"], risky)
            self.assertEqual(report["quarantined_cases"], 1)
            write_rows(inp, [{**source, "split": "test"}])
            with self.assertRaisesRegex(ValueError, "train/dev"):
                screen([inp], Path(temp) / "bad")

    def test_postjudgment_inputs_are_screened_without_deleting_ordinary_allegations(self):
        self.assertIn("postjudgment_input_review", input_risks("经二审查明，上诉人否认一审认定的事实。"))
        self.assertEqual(input_risks("公诉机关指控甲收取款项，甲如实供述。"), [])
        self.assertIn("possible_outcome_text_review", input_risks("建议在××期七年左右量刑。"))

    def test_integrity_and_review_are_distinct(self):
        source, annotated = pair()
        report, queue = audit([source], [annotated], PROFILE)
        self.assertEqual(report["original_input_label_and_split_integrity"], "passed")
        self.assertIsNone(report["semantic_accuracy"])
        self.assertFalse(report["training_ready"])
        self.assertNotIn("sentence_months", queue[0])

    def test_label_and_quote_tampering_fail(self):
        source, annotated = pair()
        changed = copy.deepcopy(annotated)
        changed["sentence_months"] += 1
        with self.assertRaisesRegex(ValueError, "label/split"):
            audit([source], [changed], PROFILE)
        changed = copy.deepcopy(annotated)
        changed["distillation"]["claims"][0]["evidence"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "offsets"):
            audit([source], [changed], PROFILE)

    def test_risk_queue_does_not_change_status(self):
        _, annotated = pair()
        annotated["distillation"]["condition_annotation"]["conditions"] = [
            {"condition_id": "returned_or_compensated", "status": "supported",
             "evidence": [{"quote": "公司已经退还款项"}]}]
        self.assertEqual(risk_flags(annotated), ["compensation_actor_review"])
        self.assertEqual(annotated["distillation"]["condition_annotation"]["conditions"][0]["status"], "supported")
