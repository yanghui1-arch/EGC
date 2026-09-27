"""Synthetic subprocess check; no model loading or API calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from egc.io import digest, read_json, read_rows, write_json, write_rows
from egc.qwen_eos_smoke import run, select_jobs


class NativeEosSmokeTests(unittest.TestCase):
    def test_bounded_run_gate_and_private_collection(self):
        jobs = []
        for i in range(12):
            messages = [{"role": "user", "content": json.dumps({"charge": str(i), "facts": "synthetic"})}]
            jobs.append({"id": str(i), "split": "dev", "messages": messages, "prompt_hash": digest(messages)})
        self.assertEqual(select_jobs(jobs + jobs), jobs)
        with self.assertRaises(ValueError):
            select_jobs(jobs[:-1])
        with self.assertRaises(ValueError):
            select_jobs([{**jobs[0], "split": "test"}] + jobs[1:])
        with self.assertRaises(ValueError):
            select_jobs([{**jobs[0], "prompt_hash": "changed"}] + jobs[1:])
        for finish in ("stop", "length"):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_json(root / "model/config.json", {"model_type": "qwen2", "eos_token_id": 151643})
                archive = root / "input.zip"
                archive.write_bytes(b"synthetic archive; unpack mocked")
                out = root / "run"
                def unpack(source, destination):
                    write_rows(destination / "direct.dev.jobs.jsonl", jobs)
                    write_json(destination / "manifest.json", {"synthetic": True})
                def execute(command, logfile):
                    if command[3] == "train":
                        self.assertEqual(command[-2:], ["--max-steps", "32"])
                        self.assertNotIn("--resume", command)
                        write_json(out / "direct/completion.json", {"complete": True, "training_mode": "lora",
                                   "global_step": 32, "artifact": str(out / "direct/adapter")})
                        write_json(out / "direct/run_manifest.json", {"chat_protocol": {"completion_end_token": "<|endoftext|>"}})
                        write_json(out / "direct/adapter/model.safetensors", {"must_not_collect": True})
                    else:
                        self.assertEqual(command[3], "infer")
                        self.assertEqual(command[command.index("--max-new-tokens") + 1], "2048")
                        self.assertEqual(read_rows(out / "probe.jobs.jsonl"), jobs)
                        settings = {"jobs_hash": digest(jobs)}
                        key = digest(settings)
                        path = out / "direct.dev.predictions.jsonl"
                        write_json(str(path) + ".manifest.json", {"settings": settings, "generation_key": key})
                        write_rows(path, [{"id": j["id"], "prompt_hash": j["prompt_hash"], "generation_key": key,
                                          "finish_reason": finish, "text": '{"reasoning":"synthetic", "sentence_months":12}'} for j in jobs])
                    return {"command": command, "exit_code": 0}
                with patch("egc.qwen_eos_smoke.platform.system", return_value="Linux"), \
                     patch("egc.qwen_eos_smoke.subprocess.check_output", return_value="synthetic-commit"), \
                     patch("egc.qwen_eos_smoke.unpack", side_effect=unpack), \
                     patch("egc.qwen_eos_smoke.execute", side_effect=execute):
                    run(archive, out, root / "model")
                result = read_json(out / "completion.json")
                self.assertEqual(result["n"], 12)
                self.assertEqual(result["termination_gate_passed"], finish == "stop")
                self.assertEqual(len(read_json(out / "execution.json")["commands"]), 2)
                with zipfile.ZipFile(out / "results.zip") as packed:
                    self.assertIn("completion.json", packed.namelist())
                    self.assertFalse(any("safetensors" in name or "jobs.jsonl" in name for name in packed.namelist()))
