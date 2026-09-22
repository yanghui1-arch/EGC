"""Synthetic offline integration: labels, caches, paired arms, packaging, metrics."""
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from egc.io import digest, read_json, read_rows, write_json, write_rows
from egc.learned import (ARMS, MEMBER, VERSION, annotate, candidate, payload, pool, prepare, validate, visible)
from egc.learned_server import (check_prediction_identity, collect, evaluate, inference_command,
                                train_command, unpack, run)


def raw(facts="甲偷取物品。乙实施诈骗。", charge="诈骗", months=72):
    return {"fact": facts, "meta": {"criminals": ["甲"], "accusation": [charge],
            "term_of_imprisonment": {"imprisonment": months, "death_penalty": False, "life_imprisonment": False}}}


def row(split="train", number=1):
    facts = "甲实施案情甲。乙实施其他案情。" if split == "train" else "甲实施验证案情。丙实施另一案情。"
    result, _ = candidate(raw(facts), str(number), {"诈骗": "诈骗"})
    result["split"] = split
    return result


def annotation(r):
    return {"decision": "keep", "issues": [], "summary": "事实仅支持原文记载的目标行为，其他情节未知。",
            "evidence": [{"quote": r["facts"].split("。")[0]+"。", "subject": "甲", "relation": "target", "kind": "action"}]}


def fixture(root):
    data = root / "pool"
    rows = [row(), row("dev", 2)]
    for split, r in zip(("train", "dev"), rows):
        write_rows(data / f"{split}.jsonl", [r])
    write_json(data / "manifest.json", {"version": VERSION, "train_hash": digest(rows[:1]),
        "dev_hash": digest(rows[1:]), "held_input_hash": digest([])})
    return data, rows


class LearnedTests(unittest.TestCase):
    def test_no_keyword_charge_inference_or_teacher_label_input(self):
        r, reason = candidate(raw(), "x", {"诈骗": "诈骗"})
        self.assertIsNone(reason)
        self.assertEqual(r["charge"], "诈骗")
        changed = dict(r, sentence_months=199, opinion="SECRET", relevant_articles_label=[999])
        self.assertEqual(payload(r, "deepseek-flash"), payload(changed, "deepseek-flash"))
        user = json.loads(payload(changed, "deepseek-flash")["messages"][1]["content"])
        self.assertEqual(set(user), {"facts", "charge", "target_person"})
        self.assertEqual(user["facts"], r["facts"])
        for months in (True, 0, -1, 2.5):
            self.assertIsNone(candidate(raw(months=months), "x", {"诈骗": "诈骗"})[0])

    def test_source_span_subject_and_uncertainty_validation(self):
        r = row()
        self.assertEqual(validate(annotation(r), r), annotation(r))
        for field, value in (("quote", "杜撰证据"), ("subject", "乙"), ("subject", None), ("relation", "guilty")):
            bad = annotation(r)
            bad["evidence"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate(bad, r)
        unknown = annotation(r)
        unknown["evidence"][0].update(subject=None, relation="uncertain")
        validate(unknown, r)
        bad = annotation(r)
        bad["sentence_months"] = 2
        with self.assertRaises(ValueError):
            validate(bad, r)

    def test_official_zip_pool_on_synthetic_source_and_wrapped_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = random.Random(42)
            cases = [raw("甲"+"".join(chr(rng.randrange(0x4e00, 0x8fff)) for _ in range(80))) for i in range(15)]
            bundle = root / "source.zip"
            with zipfile.ZipFile(bundle, "w") as z:
                z.writestr(MEMBER, "\n".join(json.dumps(r, ensure_ascii=False) for r in cases))
            held = row("test")
            write_rows(root / "benchmark.jsonl", [held])
            excluded, _ = candidate(cases[0], "old", {"诈骗": "诈骗"})
            write_rows(root / "old.jsonl", [{"row": excluded, "reason": "old"}])
            with patch("egc.learned.ARCHIVE_SHA", hashlib.sha256(bundle.read_bytes()).hexdigest()):
                report = pool(bundle, [root / "benchmark.jsonl"], [root / "old.jsonl"], root / "pool", per_charge=10)
            self.assertEqual(report["train"] + report["dev"], 10)
            retained = read_rows(root / "pool/train.jsonl") + read_rows(root / "pool/dev.jsonl")
            self.assertNotIn(excluded["facts"], [r["facts"] for r in retained])
            self.assertTrue(all(r["sentence_months"] == 72 for r in retained))

    def test_cached_annotation_pairing_unpack_and_cpu_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, rows = fixture(root)
            def teacher(request, key, **kwargs):
                facts = json.loads(request["messages"][1]["content"])["facts"]
                r = next(r for r in rows if r["facts"] == facts)
                return annotation(r), {"prompt_tokens": 20}, {"model": "resolved-version"}
            cache = root / "annotations"
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-placeholder"}), patch("egc.learned.call_deepseek", side_effect=teacher) as call:
                dry = annotate(data, cache, dry_run=True)
                self.assertEqual(dry["new_requests"], 2)
                self.assertFalse(cache.exists())
                annotate(data, cache, limit=1)
                with self.assertRaisesRegex(ValueError, "not complete"):
                    prepare(data, cache, root / "early")
                annotate(data, cache)
                annotate(data, cache)
                self.assertEqual(call.call_count, 2)
            report = prepare(data, cache, root / "ready")
            self.assertEqual(report["retained"], {"train": 1, "dev": 1})
            unpack(root / "ready/experiment.zip", root / "unpacked")
            for arm in ARMS:
                sample = read_rows(root / "unpacked" / f"{arm}.train.sft.jsonl")[0]
                answer = json.loads(sample["completion"][0]["content"])
                self.assertEqual(answer["sentence_months"], 72)
                self.assertEqual(answer["reasoning"], annotation(rows[0])["summary"])
                if arm == "direct":
                    self.assertNotIn("evidence", answer)
                else:
                    self.assertEqual(len(answer["evidence"]), 1)
            for arm in ("frozen",) + ARMS:
                jobs = read_rows(root / "unpacked" / f"{'direct' if arm == 'frozen' else arm}.dev.jobs.jsonl")
                settings = {"jobs_hash": digest(jobs)}
                pred = {"id": jobs[0]["id"], "prompt_hash": jobs[0]["prompt_hash"], "generation_key": digest(settings),
                        "finish_reason": "stop", "text": json.dumps({"reasoning": "测试", "sentence_months": 70})}
                path = root / "run" / f"{arm}.dev.predictions.jsonl"
                write_rows(path, [pred])
                write_json(str(path)+".manifest.json", {"settings": settings, "generation_key": digest(settings)})
            metrics = evaluate(root / "unpacked", root / "run")
            self.assertEqual(metrics["metrics"]["bound"]["mae_months_full"], 2)
            self.assertEqual(metrics["metrics"]["train_median"]["mae_months_full"], 0)
            self.assertEqual(metrics["comparisons"]["bound_vs_flat"]["delta_mae_candidate_minus_baseline"], 0)
            write_json(root / "run/bound/model/config.json", {"never_package": True})
            with zipfile.ZipFile(collect(root / "run")) as z:
                self.assertNotIn("bound/model/config.json", z.namelist())
            with self.assertRaises(ValueError):
                unpack(root / "ready/experiment.zip", root / "unpacked")

    def test_failed_or_uncertain_requests_never_silently_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, rows = fixture(root)
            out = root / "annotations"
            request_hash = digest(payload(rows[1], "deepseek-flash"))
            write_json(out / "cache" / f"{request_hash}.pending.json", {"id": rows[1]["id"]})
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-placeholder"}), patch("egc.learned.call_deepseek", side_effect=RuntimeError("private")) as call:
                annotate(data, out)
                report = annotate(data, out, resolve_pending_as_failed=True)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(report["status"]["failed"], 2)
            saved = "".join(p.read_text(encoding="utf-8") for p in (out / "cache").glob("*.json"))
            self.assertNotIn("private", saved)
            self.assertNotIn("test-placeholder", saved)

    def test_server_commands_and_provenance(self):
        cmd = train_command("data", "run", "bound", "model", 42, 3, 8192)
        self.assertIn("--training-mode", cmd)
        self.assertEqual(cmd[cmd.index("--checkpoint-selection")+1], "final")
        infer = inference_command("model", "jobs", "out", 8192, 42, "adapter")
        self.assertIn("adapter", infer)
        with self.assertRaisesRegex(ValueError, "identity"):
            check_prediction_identity([], [], {"settings": {"jobs_hash": "bad"}, "generation_key": "bad"})

    def test_full_server_dispatch_resume_and_failure_collection_with_cpu_fakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, rows = fixture(root)
            def teacher(request, key, **kwargs):
                facts = json.loads(request["messages"][1]["content"])["facts"]
                return annotation(next(r for r in rows if r["facts"] == facts)), {}, {}
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "placeholder"}), patch("egc.learned.call_deepseek", side_effect=teacher):
                annotate(data, root / "annotations")
            prepare(data, root / "annotations", root / "ready")
            calls = []
            def fake_execute(cmd, logfile):
                calls.append(cmd)
                out = Path(cmd[cmd.index("--output")+1])
                if cmd[3] == "train":
                    artifact = out / "model"
                    artifact.mkdir(parents=True)
                    write_json(out / "completion.json", {"complete": True, "training_mode": "full", "artifact": str(artifact)})
                else:
                    jobs = read_rows(cmd[cmd.index("--jobs")+1])
                    settings = {"jobs_hash": digest(jobs), "model": cmd[cmd.index("--model")+1],
                        "adapter": None, "seed": 42, "temperature": 0.0, "dtype": "bfloat16", "max_model_len": 8192, "max_new_tokens": 2048}
                    write_rows(out, [{"id": j["id"], "prompt_hash": j["prompt_hash"], "generation_key": digest(settings),
                        "finish_reason": "stop", "text": json.dumps({"reasoning": "合成测试", "sentence_months": 72})} for j in jobs])
                    write_json(str(out)+".manifest.json", {"settings": settings, "generation_key": digest(settings)})
                return {"command": cmd, "seconds": 0, "exit_code": 0}
            with patch("egc.learned_server.platform.system", return_value="Linux"), patch("egc.learned_server.preflight"), patch("egc.learned_server.execute", side_effect=fake_execute):
                result = run(root / "ready/experiment.zip", root / "run", model=str(root / "model"))
                self.assertTrue(Path(result["results"]).exists())
                self.assertEqual(sum(cmd[3] == "train" for cmd in calls), 3)
                self.assertEqual(sum(cmd[3] == "infer" for cmd in calls), 4)
                self.assertFalse(read_json(root / "run/completion.json")["test_evaluated"])
                run(root / "ready/experiment.zip", root / "run", model=str(root / "model"), resume=True)
                self.assertEqual(sum(cmd[3] == "train" for cmd in calls), 3)
                with self.assertRaisesRegex(ValueError, "same code"):
                    run(root / "ready/experiment.zip", root / "run", model=str(root / "model"), seed=43, resume=True)
            with patch("egc.learned_server.platform.system", return_value="Linux"), patch("egc.learned_server.preflight", side_effect=ValueError("too long")):
                with self.assertRaisesRegex(ValueError, "too long"):
                    run(root / "ready/experiment.zip", root / "failed", model=str(root / "model"))
                self.assertTrue((root / "failed/results.zip").exists())
                self.assertEqual(read_json(root / "failed/failure.json")["error_type"], "ValueError")

    def test_persistent_provider_failure_stops_before_nominal_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, original = fixture(root)
            rows = [dict(original[0], id=f"case{i}", source_id=str(i), facts=f"甲记录{i}。乙另外行为{i}。") for i in range(9)]
            write_rows(data / "train.jsonl", rows)
            write_json(data / "manifest.json", {"train_hash": digest(rows), "dev_hash": digest(original[1:])})
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "placeholder"}), patch("egc.learned.call_deepseek", side_effect=RuntimeError("unavailable")) as call:
                report = annotate(data, root / "annotations", limit=10, workers=2)
            self.assertEqual(call.call_count, 4)
            self.assertTrue(report["stopped_on_repeated_failures"])
            self.assertEqual(report["not_yet_scheduled"], 6)


if __name__ == "__main__":
    unittest.main()
