import copy
from pathlib import Path
import tempfile
import unittest

from egc.input_audit import audit
from egc.input_review import apply_review
from egc.io import digest, read_json, write_json, write_rows
from test_fact_experiment import case


def fixture():
    a = case(1, "合成案件，建议法院判处三年至四年。")
    b = case(2, "合成案件中的被告曾被判处三年，此次交易另行发生。")
    c = case(3, "独立虚构的测试事实，甲在市场实施交易。")
    d = {**case(4, c["facts"]), "split": "dev"}
    # Same-split duplicates only may be resolved by this minimal quarantine policy.
    e = case(5, c["facts"])
    splits = {"train": [a, b, c, e], "dev": [d]}
    report, queue, pairs = audit(splits)
    plan = {"version": "synthetic-v1", "reviewer": "synthetic_test",
            "input_hashes": {s: digest(rows) for s, rows in splits.items()},
            "audit_hashes": {"report": digest(report), "queue": digest(queue), "pairs": digest(pairs)},
            "quarantine_groups": {"current_case_sentence_request": [a["id"]],
                                  "reviewed_near_duplicate": [e["id"]]},
            "dismissed_flag_kinds": {b["id"]: ["possible_sentence_hint"]},
            "duplicate_links": [{"excluded": e["id"], "representative": c["id"]}]}
    return splits, report, queue, pairs, plan


class InputReviewTests(unittest.TestCase):
    def test_partition_preserves_rows_and_unresolved_risks(self):
        args = fixture(); original = copy.deepcopy(args)
        manifest, retained, quarantined, pending = apply_review(*args)
        self.assertEqual(args, original)
        self.assertEqual(manifest["candidate_splits"], {"train": 2, "dev": 1})
        self.assertEqual(manifest["quarantined_rows"], 2)
        self.assertEqual(manifest["remaining_audited_overlap_pairs"], 1)
        self.assertEqual(len(pending), 2)  # Remaining train/dev overlap is NOT cleared.
        self.assertEqual(retained["train"][0], args[0]["train"][1])
        self.assertFalse(manifest["training_ready"])
        self.assertEqual(manifest["api_calls"], 0)

    def test_rejects_changed_data_or_review_evidence(self):
        for part in (0, 1, 2, 3):
            args = fixture()
            if part == 0: args[0]["train"][0]["sentence_months"] = 987
            elif part == 1: args[1]["total_rows"] += 1
            elif part == 2: args[2][0]["facts"] += "altered"
            else: args[3].clear()
            with self.assertRaises(ValueError): apply_review(*args)

    def test_rejects_ambiguous_decisions_and_cross_split_dedup(self):
        args = fixture(); plan = args[-1]
        plan["quarantine_groups"]["another_reason"] = [args[0]["train"][0]["id"]]
        with self.assertRaises(ValueError): apply_review(*args)
        args = fixture(); plan = args[-1]
        plan["quarantine_groups"]["reviewed_near_duplicate"] = [args[0]["dev"][0]["id"]]
        plan["duplicate_links"][0]["excluded"] = args[0]["dev"][0]["id"]
        with self.assertRaises(ValueError): apply_review(*args)

    def test_dismissing_one_flag_does_not_clear_another(self):
        args = fixture(); row = args[0]["train"][1]
        row["facts"] += "此案涉及数罪并罚。"
        report, queue, pairs = audit(args[0])
        plan = args[-1]
        plan["input_hashes"] = {s: digest(rows) for s, rows in args[0].items()}
        plan["audit_hashes"] = {"report": digest(report), "queue": digest(queue), "pairs": digest(pairs)}
        _, _, _, pending = apply_review(args[0], report, queue, pairs, plan)
        entry = next(r for r in pending if r["row"]["id"] == row["id"])
        self.assertEqual([f["kind"] for f in entry["remaining_flags"]], ["possible_multiple_charge_scope"])

    def test_cli_new_directory_only_and_preserves_sources(self):
        from egc.cli import parser, run
        splits, report, queue, pairs, plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, rows in splits.items(): write_rows(root / f"{split}.jsonl", rows)
            write_json(root / "audit/report.json", report)
            write_rows(root / "audit/review_queue.jsonl", queue)
            write_json(root / "audit/overlap_pairs.json", pairs)
            write_json(root / "decisions.json", plan)
            before = (root / "train.jsonl").read_bytes()
            args = parser().parse_args(["apply-input-review", "--train", str(root / "train.jsonl"),
                "--dev", str(root / "dev.jsonl"), "--audit-dir", str(root / "audit"),
                "--decisions", str(root / "decisions.json"), "--output-dir", str(root / "out")])
            run(args)
            self.assertEqual((root / "train.jsonl").read_bytes(), before)
            self.assertFalse(read_json(root / "out/manifest.json")["training_ready"])
            self.assertEqual(len(list((root / "out").iterdir())), 6)
            with self.assertRaises(ValueError): run(args)


if __name__ == "__main__": unittest.main()
