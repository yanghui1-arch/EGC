"""Synthetic LAIC orchestration/identity checks; no real data or model calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egc.io import digest, read_json, write_json, write_rows
from egc.learned import messages
from egc.server import SFT_TOKENIZATION_VERSION
from egc.tera_data import VERSION
from egc.tera_server import ARMS, file_hash, frozen_jobs
from egc.tera_benchmark import run


class BenchmarkTests(unittest.TestCase):
    def test_four_arms_preflight_resume_and_strict_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, model = root / "training", root / "model"
            data = source / "data"
            write_json(model / "config.json", {"max_position_embeddings": 131072})
            execution = {"version": VERSION, "model": str(model), "seed": 42, "epochs": 3, "max_length": 8192}
            write_json(source / "execution.json", execution)
            write_json(source / "completion.json", {"complete": True, "phase": "complete_dev"})
            training = [{"id": "synthetic-training"}]
            write_rows(data / "tera.train.jsonl", training)
            write_json(source / "token_budget.json", {"training_hash": digest(training), "chat_protocol": {}})
            refs = [{"id": "test-a", "split": "test", "facts": "甲实施行为。", "charge": "示例", "sentence_months": 12},
                    {"id": "test-b", "split": "test", "facts": "乙实施行为。", "charge": "示例", "sentence_months": 24}]
            jobs = [{"id": r["id"], "split": "test", "variant": "direct", "messages": messages(r, "direct"),
                     "prompt_hash": digest(messages(r, "direct"))} for r in refs]
            write_rows(data / "test0.references.jsonl", refs)
            write_rows(data / "direct.test0.jobs.jsonl", jobs)
            write_json(data / "manifest.json", {"tests": [{"name": "test0", "path_label": "laic_test.jsonl", "hash": digest(refs), "n": 2}]})
            self.assertEqual(frozen_jobs(data, "test0"), jobs)
            for arm in ARMS:
                artifact = source / arm / "artifact"
                manifest = execution | {"arm": arm, "max_steps": -1, "training_hash": digest(training),
                    "chat_protocol": {}, "tokenization_version": SFT_TOKENIZATION_VERSION}
                write_json(source / arm / "run_manifest.json", manifest)
                write_json(artifact / "module.json", manifest)
                write_json(source / arm / "completion.json", {"complete": True, "artifact": str(artifact),
                    "artifact_sha256": {"module.json": file_hash(artifact / "module.json")}})
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            commands = []
            def execute(command, log):
                commands.append(command)
                path = Path(command[command.index("--output")+1])
                settings = {"jobs_hash": digest(jobs), "split": "test0"}
                if "--validate-only" not in command:
                    write_json(str(path) + ".manifest.json", {"settings": settings, "generation_key": digest(settings)})
                    write_rows(path, [{"id": j["id"], "prompt_hash": j["prompt_hash"], "generation_key": digest(settings),
                        "text": '{"reasoning":"synthetic", "sentence_months":12}', "finish_reason": "stop"} for j in jobs])
                return {"exit_code": 0}
            out = root / "laic"
            with patch("egc.tera_benchmark.require_server"), patch("egc.tera_benchmark.tokenizer_for", return_value=(None, {})), \
                 patch("egc.tera_benchmark.prompt_features", return_value=([1, 2], {}, [])), \
                 patch("egc.tera_benchmark.execute", side_effect=execute), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                run(source, out)
                self.assertEqual(len(commands), 4)
                self.assertTrue(all(c[c.index("--split")+1] == "test0" for c in commands))
                self.assertTrue(all(c[c.index("--max-length")+1] == "32768" for c in commands))
                self.assertEqual(read_json(out / "test0.metrics.json")["metrics"]["tera"]["mae_months_full"], 6)
                self.assertTrue((out / "results.zip").exists())
                self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))
                commands.clear()
                run(source, out, resume=True)
                self.assertTrue(all("--validate-only" in c for c in commands))
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    run(source, out, max_length=8192, resume=True)
                commands.clear()
                with patch("egc.tera_benchmark.prompt_features", return_value=([1]*31000, {}, [])):
                    with self.assertRaisesRegex(ValueError, "context budget exceeded"):
                        run(source, root / "too_long")
                self.assertEqual(commands, [])
                self.assertTrue((root / "too_long/failure.json").exists())
                self.assertTrue((root / "too_long/results.zip").exists())
            altered = [dict(jobs[0], messages=[{"role": "user", "content": "gold answer"}]), jobs[1]]
            write_rows(data / "direct.test0.jobs.jsonl", altered)
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                frozen_jobs(data, "test0")
            write_rows(data / "direct.test0.jobs.jsonl", jobs)
            write_rows(data / "test0.references.jsonl", [dict(refs[0], sentence_months=999), refs[1]])
            with self.assertRaisesRegex(ValueError, "reference identity"):
                frozen_jobs(data, "test0")


if __name__ == "__main__":
    unittest.main()
