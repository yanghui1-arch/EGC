import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from egc.b0 import encoded, lines, pack_files
from egc.io import read_rows, digest
from egc.learned import prepare, annotate, messages
from egc.learned_screen import build_screened, risk_spans
from egc.learned_server import unpack
from test_learned import fixture, annotation


class ScreenTests(unittest.TestCase):
    def test_flags_are_not_semantic_labels(self):
        self.assertTrue(risk_spans("建议判处三年以下。"))
        # Expected false positive; explicitly quarantined as unresolved, not wrong.
        self.assertTrue(risk_spans("此前于2010年被判处刑罚。"))
        self.assertFalse(risk_spans("甲租了仓库。随后加工材料。"))

    def test_shared_whole_case_filter_keeps_labels_and_tests_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, rows = fixture(root)
            def teacher(request, key, **kwargs):
                body = annotation(json.loads(request["messages"][1]["content"]))
                return body, {}, {"model": "synthetic"}
            with patch("egc.learned.call_deepseek", side_effect=teacher), patch("egc.credentials.read_key", return_value="TEST_ONLY"):
                annotate(pool, root / "annotations", workers=1)
            prepare(pool, root / "annotations", root / "ready")
            original = root / "ready/experiment.zip"
            with ZipFile(original) as z:
                files = {n:z.read(n) for n in z.namelist() if n != "checksums.json"}
            # Add a second, paired synthetic training case with a numeric recommendation.
            for name in ["train.references.jsonl"]+[a+".train.sft.jsonl" for a in ("direct","flat","bound")]:
                records = [json.loads(l) for l in files[name].decode().splitlines()]
                extra = dict(records[0], id="risk-case")
                if name == "train.references.jsonl":
                    extra["facts"] = "公诉机关建议判处三年以下。"
                records.append(extra)
                files[name] = lines(records)
            manifest = json.loads(files["manifest.json"])
            manifest["retained"]["train"] = 2
            test_row = dict(rows[1], id="held-test", split="test")
            files["test0.references.jsonl"] = lines([test_row])
            manifest["tests"] = [{"name":"test0", "hash":digest([test_row]), "n":1}]
            for arm in ("direct","flat","bound"):
                prompt = messages(test_row, arm)
                files[arm+".test0.jobs.jsonl"] = lines([{"id":test_row["id"], "messages":prompt, "prompt_hash":digest(prompt)}])
            files["manifest.json"] = encoded(manifest)
            source = root / "source.zip"
            pack_files(source, files)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch("egc.learned.call_deepseek") as api:
                result = build_screened(source, root / "screened")
                api.assert_not_called()
            self.assertEqual(result["quarantined"], {"train":1})
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            unpack(root / "screened/experiment.zip", root / "unpacked")
            self.assertEqual(read_rows(root / "unpacked/train.references.jsonl"), rows[:1])
            with ZipFile(root / "screened/experiment.zip") as z:
                self.assertEqual(z.read("dev.references.jsonl"), files["dev.references.jsonl"])
                for arm in ("direct","flat","bound"):
                    self.assertEqual(z.read(arm+".dev.jobs.jsonl"), files[arm+".dev.jobs.jsonl"])
                    self.assertEqual(z.read(arm+".test0.jobs.jsonl"), files[arm+".test0.jobs.jsonl"])
                self.assertEqual(z.read("test0.references.jsonl"), files["test0.references.jsonl"])
            with self.assertRaises(ValueError):
                build_screened(source, root / "screened")
