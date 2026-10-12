"""Synthetic correction/export/field-mask integration; no real data or API."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egc.b0 import encoded, lines, pack_files
from egc.correct_training import BALANCED, TOKEN_MEAN, build
from egc.io import digest, read_json, read_rows
from egc.learned import ARMS, VERSION, messages
from egc.learned_audit import checked_zip
from egc.learned_server import unpack
from egc.tera_data import training_rows
from egc.tera_server import prepare
from test_tera import CharacterTokenizer


class CorrectionTests(unittest.TestCase):
    def test_package_preserves_cases_labels_tests_and_drives_server_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files, refs = {}, {}
            for split in ("train", "dev", "test0", "test1", "test2"):
                records = [{"id": split + str(i), "source_id": split + str(i), "dataset": "synthetic",
                            "split": split if split in ("train", "dev") else "test", "charge": "示例",
                            "facts": f'甲执行{split}{i}。乙在场，甲某某记录120。甲离开。', "opinion": None,
                            "sentence_months": m} for i, m in enumerate((12, 120) if split == "train" else (36,))]
                if split in ("train", "dev"):
                    for r in records:
                        r["target_person"] = "甲"
                refs[split] = records
                files[f"{split}.references.jsonl"] = lines(records)
                for arm in ARMS:
                    jobs, sft = [], []
                    for r in records:
                        prompt = messages(r, arm)
                        answer = {"reasoning": '引用"120"和12只是文本。', "sentence_months": r["sentence_months"]}
                        if arm != "direct":
                            evidence = [{"quote": r["facts"].split("。")[0] + "。", "relation": "target", "subject": "甲", "kind": "action"}]
                            answer = {"evidence": evidence if arm == "bound" else [x["quote"] for x in evidence]} | answer
                        sft.append({k: r[k] for k in ("id", "source_id", "dataset", "split")} |
                                   {"variant": arm, "prompt": prompt, "completion": [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]})
                        jobs.append({"id": r["id"], "split": r["split"], "variant": arm,
                                     "messages": prompt, "prompt_hash": digest(prompt)})
                    if split in ("train", "dev"):
                        files[f"{arm}.{split}.sft.jsonl"] = lines(sft)
                    if split != "train":
                        files[f"{arm}.{split}.jobs.jsonl"] = lines(jobs)
            files["manifest.json"] = encoded({"version": VERSION, "arms": list(ARMS),
                "retained": {"train": 2, "dev": 1}, "tests": [{"name": s, "n": 1, "hash": digest(refs[s])}
                                                                  for s in ("test0", "test1", "test2")]})
            source = root / "source.zip"
            pack_files(source, files)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            for balanced, loss in ((True, BALANCED), (False, TOKEN_MEAN)):
                out = root / str(balanced)
                with patch("egc.learned.call_deepseek") as api:
                    report = build(source, out, balanced)
                    api.assert_not_called()
                self.assertEqual(report["sentence_loss"], loss)
                self.assertEqual(report["splits"]["train"]["cases_with_target_mentions"], 2)
                self.assertEqual(source_hash, hashlib.sha256(source.read_bytes()).hexdigest())
                corrected = checked_zip(out / "experiment.zip")
                for name, content in files.items():
                    if "test" in name:
                        self.assertEqual(corrected[name], content)
                    if name.endswith(".sft.jsonl"):
                        before = [json.loads(x) for x in content.decode().splitlines()]
                        after = [json.loads(x) for x in corrected[name].decode().splitlines()]
                        self.assertEqual([x["completion"] for x in before], [x["completion"] for x in after])
                data = out / "data"
                unpack(out / "experiment.zip", data)
                self.assertEqual(read_rows(data / "train.references.jsonl"),
                                 [{k: v for k, v in r.items() if k != "target_person"} for r in refs["train"]])
                # Label-only sidecar: both literal 甲 mentions, never the 甲某某 placeholder of another person.
                for split in ("train", "dev"):
                    for m, r in zip(read_rows(data / f"{split}.target_supervision.jsonl"), refs[split]):
                        self.assertEqual([r["facts"][a:b] for a, b in m["target_char_spans"]], ["甲", "甲"])
                        self.assertNotIn(r["facts"].index("甲某某"), [a for a, _ in m["target_char_spans"]])
                with patch("egc.tera_server.tokenizer_for", return_value=(CharacterTokenizer(), {})):
                    budget = prepare(data, "synthetic", 8192)
                self.assertEqual(budget["sentence_loss"], loss)
                prepared = read_rows(data / "tera.train.jsonl")
                for row in prepared:
                    self.assertEqual(row["meta"]["target"], [])
                    self.assertEqual(len(row["meta"]["target_mentions"]), 2)
                    self.assertEqual({chr(row["input_ids"][i]) for i in row["meta"]["target_mentions"]}, {"甲"})
                    self.assertEqual("sentence_mask" in row, balanced)
                    if balanced:
                        selected = ''.join(chr(t) for t, on in zip(row["input_ids"], row["sentence_mask"]) if on)
                        self.assertEqual(selected, str(refs["train"][int(row["id"][-1])]["sentence_months"]))
                with self.assertRaises(ValueError):
                    build(source, out, balanced)
                with self.assertRaises(ValueError):
                    build(out / "experiment.zip", root / "recorrect")
            # Corruption must fail before the new output is created.
            bad = dict(files)
            sft = [json.loads(x) for x in bad["direct.train.sft.jsonl"].decode().splitlines()]
            answer = json.loads(sft[0]["completion"][0]["content"])
            sft[0]["completion"][0]["content"] = json.dumps(answer | {"sentence_months": 999}, ensure_ascii=False)
            bad["direct.train.sft.jsonl"] = lines(sft)
            pack_files(root / "bad.zip", bad)
            with self.assertRaisesRegex(ValueError, "gold label"):
                build(root / "bad.zip", root / "bad-output")
            self.assertFalse((root / "bad-output").exists())

    def test_digit_mask_covers_only_real_number_and_rejects_changed_offsets(self):
        tokenizer = CharacterTokenizer()
        visible = {"facts": '甲记录120。乙说"12"。', "charge": "示例"}
        for month in (0, 7, 12, 120):
            answer = {"reasoning": '摘要有120、12和"sentence_months"。', "sentence_months": month}
            direct = {"id": "synthetic", "split": "train", "prompt": messages(visible, "direct"),
                      "completion": [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]}
            bound = copy.deepcopy(direct)
            bound["completion"][0]["content"] = json.dumps({"evidence": [{"quote": "甲记录120。", "relation": "target"}]} | answer, ensure_ascii=False)
            prepared, _ = training_rows(tokenizer, {}, [direct], [bound], 8192, True)
            row = prepared[0]
            self.assertEqual(''.join(chr(t) for t, on in zip(row["input_ids"], row["sentence_mask"]) if on), str(month))
            self.assertFalse(any(row["sentence_mask"][:row["meta"]["prompt_length"]]))
            self.assertNotIn("sentence_mask", training_rows(tokenizer, {}, [direct], [bound], 8192)[0][0])
        class BadOffsets(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                encoded = super().__call__(text, **kwargs)
                if '"sentence_months": 120}' in text:
                    encoded["offset_mapping"] = [(0, 0)] * len(encoded["input_ids"])
                return encoded
        with self.assertRaisesRegex(ValueError, "token span"):
            training_rows(BadOffsets(), {}, [direct], [bound], 8192, True)
