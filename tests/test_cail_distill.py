import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from egc.cail import BenchmarkGuard, adapt, build, charge_key
from egc.distill import payload, run, validate
from egc.annotate import call_deepseek
from egc.io import read_json, read_rows, write_rows
from test_pipeline import row


def source():
    return {"fact": "被告人实施了合成行为A，其他事实详见此独立材料。", "meta": {
        "accusation": ["诈骗"], "criminals": ["甲"], "relevant_articles": [266],
        "term_of_imprisonment": {"death_penalty": False, "life_imprisonment": False, "imprisonment": 36}}}


PROFILE = {"version": "fixture-v1", "conditions": [{"id": "a", "text": "实施合成行为A"}]}
BODY = {"claims": [{"text": "材料明确记载实施行为A。", "quotes": ["实施了合成行为A"]}],
        "conditions": [{"condition_id": "a", "status": "supported", "quotes": ["实施了合成行为A"]}]}


class CailTests(unittest.TestCase):
    def test_original_labels_and_structured_charge_names(self):
        result, reason = adapt(source(), "train:1", {"诈骗": "诈骗罪"})
        self.assertIsNone(reason)
        self.assertEqual(result["sentence_months"], 36)
        self.assertIsNone(result["opinion"])
        self.assertEqual(charge_key("非法[制造、买卖、运输、邮寄、储存][枪支、弹药、爆炸物]"),
                         charge_key("非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪"))

    def test_ambiguous_or_special_labels_excluded(self):
        for field, value in (("death_penalty", True), ("life_imprisonment", True), ("imprisonment", 0),
                             ("imprisonment", True), ("imprisonment", "36")):
            raw = source()
            raw["meta"]["term_of_imprisonment"][field] = value
            self.assertIsNone(adapt(raw, "train:1", {"诈骗": "诈骗罪"})[0])
        raw = source()
        raw["meta"]["accusation"].append("抢劫")
        self.assertIsNone(adapt(raw, "train:1", {"诈骗": "诈骗罪"})[0])
        raw = source()
        raw["fact"] += "公诉机关建议判处十五年。"
        self.assertEqual(adapt(raw, "train:1", {"诈骗": "诈骗罪"})[1], "possible_outcome_text_review")

    def test_exact_and_containment_overlap(self):
        fact = "".join(chr(0x4e00+i) for i in range(100))
        guard = BenchmarkGuard([row("test", facts=fact)])
        self.assertEqual(guard.match(" \n" + fact)["method"], "normalized_exact")
        self.assertEqual(guard.match(fact + "额外案情")["method"], "character_5gram_containment")
        self.assertIsNone(guard.match("此材料与测试事实无关。"))

    def test_contest_train_member_allowed_test_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(root / "benchmark.jsonl", [dict(row("test", facts="独立评测材料"), charge="诈骗罪")])
            raws = []
            for i in range(10):
                raw = source()
                raw["fact"] = f"独立训练样本{i}，此处明确说明具体案情{i}。"
                raws.append(raw)
            name = "final_all_data/exercise_contest/data_train.json"
            with zipfile.ZipFile(root / "cail.zip", "w") as archive:
                archive.writestr(name, "\n".join(json.dumps(r, ensure_ascii=False) for r in raws))
            result = build(root / "cail.zip", name, [root / "benchmark.jsonl"], root / "pool", 10)
            self.assertEqual(result["train"]+result["dev"], 10)
            with self.assertRaises(ValueError):
                build(root / "cail.zip", "data_test.json", [], root / "bad")


class DistillTests(unittest.TestCase):
    def test_curl_credentials_only_in_stdin_and_response_metadata(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(BODY)}}],
                    "usage": {"prompt_tokens": 12}, "model": "deepseek-flash"}
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(response)+"\n200")) as process:
            body, usage, meta = call_deepseek({"model": "deepseek-flash"}, "fake-test-key", True, "curl")
            self.assertEqual(body, BODY)
            self.assertEqual(meta["model"], "deepseek-flash")
            self.assertNotIn("fake-test-key", repr(process.call_args.args))
            self.assertIn("fake-test-key", process.call_args.kwargs["input"])
            self.assertNotIn("shell", process.call_args.kwargs)

    def test_network_failure_recorded_and_not_retried(self):
        rows = [dict(row(), opinion=None)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}), \
                patch("egc.distill.call_deepseek", side_effect=RuntimeError("connection closed")) as call:
            self.assertEqual(run(rows, PROFILE, tmp)["rejected"], 1)
            self.assertTrue(read_json(Path(tmp)/"manifest.json")["complete"])
            run(rows, PROFILE, tmp)
            self.assertEqual(call.call_count, 1)

    def test_teacher_cannot_see_labels_or_reference(self):
        original = row()
        changed = dict(original, sentence_months=999, opinion="SECRET GOLD", relevant_articles_label=[1])
        self.assertEqual(payload(original, PROFILE, "deepseek-flash", 4096), payload(changed, PROFILE, "deepseek-flash", 4096))
        self.assertEqual(payload(original, PROFILE, "deepseek-flash", 4096)["thinking"], {"type": "disabled"})

    def test_evidence_validation_and_no_gold_claim(self):
        result = validate(BODY, row(), PROFILE)
        self.assertFalse(result["human_reviewed"])
        bad = copy.deepcopy(BODY)
        bad["claims"][0]["quotes"] = ["输入未记载的情况"]
        with self.assertRaises(ValueError):
            validate(bad, row(), PROFILE)

    def test_cache_limit_preserves_original_labels(self):
        rows = [dict(row(number=i), opinion=None) for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}), \
                patch("egc.distill.call_deepseek", return_value=(BODY, {"prompt_tokens": 10}, {"model": "fixture"})) as call:
            result = run(rows, PROFILE, tmp, limit=1)
            self.assertEqual(result["new_successful_api_calls"], 1)
            self.assertEqual(read_rows(Path(tmp)/"accepted.jsonl")[0]["sentence_months"], 73)
            run(rows, PROFILE, tmp, limit=1)
            self.assertEqual(call.call_count, 2)
            run(rows, PROFILE, tmp, limit=1)
            self.assertEqual(call.call_count, 2)
            self.assertTrue(read_json(Path(tmp)/"manifest.json")["complete"])

    def test_test_and_existing_court_reasoning_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rows in ([dict(row("test"), opinion=None)], [row()]):
                with self.assertRaises(ValueError):
                    run(rows, PROFILE, tmp, dry_run=True)

    def test_rejected_response_cached_without_rebilling(self):
        bad = copy.deepcopy(BODY)
        bad["claims"][0]["quotes"] = ["不存在的片段"]
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}), \
                patch("egc.distill.call_deepseek", return_value=(bad, {}, {})) as call:
            rows = [dict(row(), opinion=None)]
            self.assertEqual(run(rows, PROFILE, tmp)["rejected"], 1)
            self.assertEqual(run(rows, PROFILE, tmp)["rejected"], 1)
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
