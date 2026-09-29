"""Synthetic, offline checks only: no API, pretrained checkpoint or benchmark data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egc.io import digest, read_json, write_json, write_rows
from egc.learned import messages
from egc.tera_data import prompt_features, sentence_spans, training_rows, weak_roles


class CharacterTokenizer:
    is_fast = True
    def apply_chat_template(self, rows, tokenize, add_generation_prompt, enable_thinking):
        text = "".join("<" + r["role"] + ">" + r["content"] + "<end>" for r in rows)
        return text + ("<assistant>" if add_generation_prompt else "")
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i+1) for i in range(len(text))]}


class DataTests(unittest.TestCase):
    def test_escaping_missing_target_and_sparse_roles(self):
        tokenizer = CharacterTokenizer()
        row = {"facts": '甲说"我叫甲"。\n乙写\\字。', "charge": "示例", "target_person": "甲"}
        prompt = messages(row, "direct")
        ids, meta, spans = prompt_features(tokenizer, prompt)
        rendered = "".join(map(chr, ids))
        memory = ["".join(rendered[i] for i in span) for span in meta["sentences"]]
        self.assertEqual(memory, [json.dumps(row["facts"][a:b], ensure_ascii=False)[1:-1] for a, b in spans])
        self.assertEqual("".join(rendered[i] for i in meta["target"]), "甲")
        absent = dict(row)
        absent.pop("target_person")
        self.assertEqual(prompt_features(tokenizer, messages(absent, "direct"))[1]["target"], [])
        evidence = [{"quote": '甲说"我叫甲"。', "relation": "target"},
                    {"quote": "乙写\\字。", "relation": "other"}]
        roles, stats = weak_roles(row["facts"], spans, evidence)
        self.assertEqual(roles, [0, 1])
        self.assertEqual(stats["aligned_quotes"], 2)
        roles, stats = weak_roles("甲。乙。丙。", sentence_spans("甲。乙。丙。"), [
            {"quote": "甲。", "relation": "target"}, {"quote": "甲。", "relation": "other"},
            {"quote": "乙。丙。", "relation": "uncertain"}])
        self.assertEqual(roles, [-100, -100, -100])
        self.assertEqual(stats["conflict"], 1)
        self.assertEqual(stats["cross_boundary_quote"], 1)
        self.assertEqual(weak_roles("甲。甲。", sentence_spans("甲。甲。"), [{"quote": "甲。", "relation": "target"}])[0], [-100, -100])
        with self.assertRaises(KeyError):
            weak_roles("甲。", [(0, 2)], [{"quote": "甲。"}])

    def test_same_cases_and_no_answer_features(self):
        visible = {"facts": "甲实施案情。乙实施他案。", "charge": "示例", "target_person": "甲"}
        answer = {"reasoning": "说明", "sentence_months": 12}
        direct = {"id": "x", "split": "train", "prompt": messages(visible, "direct"),
                  "completion": [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]}
        bound = dict(direct, prompt=messages(visible, "bound"), completion=[{"role": "assistant", "content": json.dumps(
            answer | {"evidence": [{"quote": "甲实施案情。", "relation": "target"}]}, ensure_ascii=False)}])
        rows, report = training_rows(CharacterTokenizer(), {}, [direct], [bound], 8192)
        self.assertEqual(rows[0]["role_labels"], [0, -100])
        self.assertEqual(report["cases"], 1)
        altered = copy.deepcopy(direct)
        altered["completion"][0]["content"] = '{"reasoning":"changed", "sentence_months":999}'
        self.assertEqual(prompt_features(CharacterTokenizer(), direct["prompt"]), prompt_features(CharacterTokenizer(), altered["prompt"]))
        with self.assertRaises(ValueError):
            training_rows(CharacterTokenizer(), {}, [altered], [bound], 8192)


class OrchestrationTests(unittest.TestCase):
    def test_resume_revalidates_saved_arms_retrains_only_missing_and_preserves_failure(self):
        from egc.tera_server import run
        from egc.tera_data import VERSION
        from egc.server import SFT_TOKENIZATION_VERSION
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            archive = root / "input.zip"
            archive.write_bytes(b"synthetic")
            out = root / "run"
            execution = {"version": VERSION, "model": str(root), "seed": 42, "epochs": 3,
                "max_length": 8192, "archive_sha256": hashlib.sha256(b"synthetic").hexdigest(), "smoke_only": False}
            write_json(out / "execution.json", execution)
            write_json(out / "failure.json", {"old_failure": True})
            write_rows(out / "data/tera.train.jsonl", [{"id": "synthetic"}])
            training_hash = digest([{"id": "synthetic"}])
            write_json(out / "token_budget.json", {"training_hash": training_hash, "chat_protocol": {}})
            for name in ("smoke", "direct", "generic", "tera_noaux"):
                write_json(out / name / "run_manifest.json", execution | {"arm": "tera" if name == "smoke" else name,
                    "max_steps": 32 if name == "smoke" else -1, "training_hash": training_hash,
                    "chat_protocol": {}, "tokenization_version": SFT_TOKENIZATION_VERSION})
                if name != "tera_noaux":
                    write_json(out / name / "completion.json", {"complete": True})
                    write_rows(out / f"{name}.dev.predictions.jsonl", [{"id": str(i),
                        "text": '{"reasoning":"ok","sentence_months":12}', "finish_reason": "stop"} for i in range(12)])
            original_predictions = (out / "direct.dev.predictions.jsonl").read_bytes()
            commands = []
            with patch("egc.tera_server.require_server"), patch("egc.learned_audit.checked_zip", return_value={}), \
                 patch("egc.tera_server.execute", side_effect=lambda command, log: commands.append(command) or {"exit_code": 0}), \
                 patch("egc.tera_server.evaluate"), patch("egc.tera_server.collect", return_value="synthetic.zip"), \
                 patch("egc.tera_server.prepare") as prepare, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
                run(archive, out, model=root, resume=True, skip_laic=True)
                self.assertEqual([c[c.index("--arm")+1] for c in commands if c[3] == "train"], ["tera_noaux", "tera"])
                self.assertEqual(sum("--validate-only" in c for c in commands), 3)
                prepare.assert_not_called()
                self.assertEqual((out / "direct.dev.predictions.jsonl").read_bytes(), original_predictions)
                self.assertEqual(len(list(out.glob("failed_tera_noaux_*/run_manifest.json"))), 1)
                self.assertEqual(len(list(out.glob("failure_before_resume_*.metrics.json"))), 1)
                commands.clear()
                with self.assertRaisesRegex(ValueError, "mismatch: seed"):
                    run(archive, out, model=root, seed=7, resume=True)
                self.assertEqual(commands, [])

    def test_readonly_diagnostic_inventory_fixed_inputs_and_no_training(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from egc.tera_server import diagnose_cache, file_hash
        from egc.tera_data import VERSION
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            execution = {"version": VERSION, "model": str(root / "base"), "seed": 42, "epochs": 3, "max_length": 8192}
            write_json(root / "execution.json", execution)
            jobs = [{"id": str(i), "split": "dev", "messages": [{"role": "user", "content": "synthetic"}],
                     "prompt_hash": digest([{"role": "user", "content": "synthetic"}])} for i in range(5)]
            write_rows(root / "data/direct.dev.jobs.jsonl", jobs)
            write_json(root / "token_budget.json", {"dev_jobs_hash": digest(jobs), "training_hash": "synthetic", "chat_protocol": {}})
            artifact = root / "direct/artifact"
            write_json(artifact / "module.json", execution | {"arm": "direct", "training_hash": "synthetic", "chat_protocol": {}, "max_steps": -1})
            write_json(root / "direct/completion.json", {"artifact": str(artifact), "complete": True,
                "global_step": 894, "training_mode": "lora", "artifact_sha256": {"module.json": file_hash(artifact / "module.json")}})
            before = file_hash(artifact / "module.json")
            with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None),
                                             backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))),
                                             "transformers": SimpleNamespace(set_seed=lambda seed: None)}), \
                 patch("egc.tera_server.require_server"), patch("egc.tera_server.tokenizer_for", return_value=(None, {})), \
                 patch("egc.tera_server.prompt_features", return_value=([1, 2], {}, [])), \
                 patch("egc.tera_server.load_model", return_value=(Mock(), 1, "lora")) as load, \
                 patch("egc.tera_server.cache_probe", return_value={"passed": False}) as probe, \
                 patch("egc.tera_server.collect", return_value="synthetic.zip"), patch("egc.tera_server.train") as train:
                report = diagnose_cache(root)
            self.assertEqual(report["selected_ids"], ["0", "1", "2"])
            self.assertEqual(probe.call_count, 4)
            self.assertEqual(probe.call_args.kwargs["precision"], "fp32")
            load.assert_called_once()
            train.assert_not_called()
            self.assertEqual(report["models"]["direct"]["status"], "diagnosed")
            self.assertEqual(report["models"]["tera_noaux"]["status"], "no_exported_checkpoint_to_probe")
            self.assertFalse(report["training_performed"])
            self.assertEqual(file_hash(artifact / "module.json"), before)

    def test_gate_fresh_full_runs_and_failure_collection(self):
        from egc.tera_server import ARMS, run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "input.zip"
            archive.write_bytes(b"synthetic")
            commands = []
            def execute(command, log):
                commands.append(command)
                if command[2] == "egc.tera_benchmark":
                    return {"exit_code": 0}
                out = Path(command[command.index("--output") + 1])
                if command[3] == "infer":
                    write_rows(out, [{"id": str(i), "text": '{"reasoning":"ok","sentence_months":12}', "finish_reason": "stop"} for i in range(12)])
                return {"exit_code": 0}
            with patch("egc.tera_server.require_server"), patch("egc.tera_server.unpack"), patch("egc.tera_server.prepare"), \
                 patch("egc.tera_server.execute", side_effect=execute), patch("egc.tera_server.evaluate") as score, \
                 patch("egc.tera_server.collect", return_value="synthetic_results.zip"), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
                run(archive, root / "run", model=root)
                self.assertEqual(len(commands), 11)
                self.assertEqual(commands[-1][2], "egc.tera_benchmark")
                self.assertIn("32", commands[0])
                self.assertEqual([c[c.index("--arm")+1] for c in commands[2:10:2]], list(ARMS))
                self.assertTrue(all("-1" in c for c in commands[2:10:2]))
                score.assert_called_once()
                with patch("egc.tera_server.read_rows", return_value=[{"text": "bad", "finish_reason": "length"}]):
                    with self.assertRaisesRegex(ValueError, "Smoke termination"):
                        run(archive, root / "failed", model=root)
                self.assertEqual(read_json(root / "failed/failure.json")["phase"], "infer_smoke")
                self.assertFalse((root / "failed/completion.json").exists())


try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU tensor checks need torch + transformers; stdlib checks still run")
class ModelTests(unittest.TestCase):
    def setUp(self):
        from egc.tera_model import MemoryCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(42)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))
        self.model = MemoryCausalLM(base, "tera", rank=8)
        self.meta = {"prompt_length": 8, "sentences": [[1, 2], [3, 4]], "charge": [0], "target": [6]}
        self.ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])

    def test_zero_init_gradients_causality_cache_and_reload(self):
        model = self.model.eval()
        with torch.no_grad():
            h = model.base.model(self.ids, use_cache=False).last_hidden_state[0]
            memory = model.adapter.memory(h, self.meta)
            torch.testing.assert_close(model.adapter(h[7:], memory), h[7:], rtol=0, atol=0)
        batch = {"input_ids": self.ids, "labels": torch.tensor([[-100]*8 + [13, 14, 15]]),
                 "meta": self.meta, "role_labels": [0, 1]}
        loss = model(**batch)["loss"]
        loss.backward()
        self.assertGreater(model.adapter.up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.adapter.role.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.base.model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item(), 0)
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        model.eval()
        with torch.no_grad():
            h = model.base.model(self.ids, use_cache=False).last_hidden_state[0]
            changed = self.ids.clone()
            changed[0, 8:] = torch.tensor([31, 32, 33])
            altered = model.base.model(changed, use_cache=False).last_hidden_state[0]
            memory = model.adapter.memory(h, self.meta)
            other = model.adapter.memory(altered, self.meta)
            for k in memory:
                torch.testing.assert_close(memory[k], other[k], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(model.adapter(h[7:8], memory), model.adapter(altered[7:8], other), atol=1e-6, rtol=1e-5)
            logits, state, past = model.step(self.ids[:, :8], meta=self.meta)
            torch.testing.assert_close(logits, model.base.lm_head(model.adapter(h[7:8], memory))[0], atol=1e-6, rtol=1e-5)
            for pos in range(8, 11):
                logits, state, past = model.step(self.ids[:, pos:pos+1], memory=state, past=past)
                expected = model.base.lm_head(model.adapter(h[pos:pos+1], memory))[0]
                torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "module.pt"
                torch.save(model.adapter.state_dict(), path)
                model.adapter.load_state_dict(torch.load(path, weights_only=True), strict=True)
                torch.testing.assert_close(model.adapter(h[7:8], model.adapter.memory(h, self.meta)), model.adapter(h[7:8], memory))
        bad = dict(self.meta, sentences=[[8]])
        with self.assertRaises(ValueError):
            model.adapter.memory(h, bad)
        with self.assertRaises(ValueError):
            model.step(self.ids.expand(2, -1), meta=self.meta)

    def test_cache_probe_reports_backbone_and_keeps_failed_gate(self):
        from egc.tera_server import cache_probe
        model = self.model.eval()
        report = cache_probe(model, self.ids[0, :8].tolist(), self.meta, steps=4)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["steps"]), 4)
        self.assertLess(report["steps"][0]["module_increment_difference_max_abs"], 1e-6)
        original = model.step
        def drift(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("return_base_logits"):
                logits, memory, past, baseline = result
                # Uniform shift preserves greedy tokens but must still fail the old .25 gate.
                return logits + 0.3125, memory, past, baseline
            return result
        with patch.object(model, "step", side_effect=drift):
            report = cache_probe(model, self.ids[0, :8].tolist(), self.meta)
        self.assertFalse(report["passed"])
        self.assertTrue(report["argmax_equal"])
        self.assertGreater(report["steps"][0]["module_increment_difference_max_abs"], 0.3)
        self.assertLess(report["steps"][0]["backbone_max_abs"], 1e-5)

    def test_precision_validation_checks_fp32_and_restores_mixed_dtypes_on_error(self):
        from egc.tera_server import validate_cache
        model = self.model.eval()
        model.backbone.bfloat16()
        model.register_buffer("synthetic_bf16_buffer", torch.ones(2, dtype=torch.bfloat16))
        before = {n: t.clone() for n, t in model.state_dict().items()}
        bf16 = {"passed": False, "argmax_equal": True, "memory_max_abs": 0.01, "cached_vs_full_max_abs": 0.4375}
        fp32 = {"passed": True, "argmax_equal": True, "memory_max_abs": 0., "cached_vs_full_max_abs": 0.0001}
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        for control, accepted in ((fp32, True), (fp32 | {"cached_vs_full_max_abs": .01}, False)):
            with patch("egc.tera_server.cache_probe", side_effect=[bf16, control]) as probe:
                report = validate_cache(model, [], {})
            self.assertEqual(report["passed"], accepted)
            self.assertFalse(report["legacy_absolute_gate_passed"])
            self.assertEqual(probe.call_args.kwargs, {"steps": 4, "precision": "fp32"})
        with patch("egc.tera_server.cache_probe", side_effect=[bf16 | {"argmax_equal": False}, fp32]):
            self.assertFalse(validate_cache(model, [], {})["passed"])
        with patch("egc.tera_server.cache_probe", side_effect=[bf16, RuntimeError("synthetic failure")]):
            with self.assertRaises(RuntimeError):
                validate_cache(model, [], {})
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_tf32)
        for n, tensor in model.state_dict().items():
            self.assertEqual(tensor.dtype, before[n].dtype)
            self.assertTrue(torch.equal(tensor, before[n]))
        # The actual tiny model exercises the FP32 control, not only mocked reports.
        model.float()
        self.assertTrue(validate_cache(model, self.ids[0, :8].tolist(), self.meta)["passed"])

    def test_saved_prediction_revalidation_is_readonly_and_checks_identity(self):
        from egc.tera_server import infer, cache_probe, file_hash
        from egc.tera_data import VERSION
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data, trained = root / "data", root / "direct"
            artifact = trained / "artifact"
            ids = self.ids[0, :8].tolist()
            self.model.config.max_position_embeddings = 32768
            probe = cache_probe(self.model, ids, self.meta)
            manifest = {"version": VERSION, "model": str(root), "arm": "direct", "chat_protocol": {},
                        "training_hash": "synthetic", "seed": 42, "max_length": 8192}
            write_json(artifact / "module.json", manifest)
            checks = {"module.json": file_hash(artifact / "module.json")}
            write_json(trained / "completion.json", {"complete": True, "artifact": str(artifact),
                "artifact_sha256": checks, "training_mode": "lora", "cache_probe": probe})
            jobs = [{"id": "x", "messages": [], "prompt_hash": digest([])}]
            write_rows(data / "direct.dev.jobs.jsonl", jobs)
            write_json(root / "token_budget.json", {"training_hash": "synthetic", "dev_jobs_hash": digest(jobs)})
            settings = {"version": VERSION, "model": str(root), "arm": "direct", "artifact_hash": digest(checks),
                "jobs_hash": digest(jobs), "training_hash": "synthetic", "backend": "transformers_request_local_cache",
                "seed": 42, "temperature": 0, "max_new_tokens": 2048, "max_model_len": 8192,
                "dtype": "bfloat16", "chat_protocol": {}}
            output = root / "direct.dev.predictions.jsonl"
            write_rows(output, [{"id": "x", "prompt_hash": digest([]), "generation_key": digest(settings)}])
            sidecar = str(output) + ".manifest.json"
            write_json(sidecar, {"settings": settings, "generation_key": digest(settings)})
            before = {p: p.read_bytes() for p in (output, Path(sidecar), trained / "completion.json", artifact / "module.json")}
            with patch("egc.tera_server.require_server"), patch("egc.tera_server.tokenizer_for", return_value=(None, {})), \
                 patch("egc.tera_server.prompt_features", return_value=(ids, self.meta, [])), \
                 patch("egc.tera_server.load_model", return_value=(self.model, 1, "lora")):
                infer(data, trained, output, root, 42, 8192, validate_only=True)
                self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))
                self.assertEqual(len(list(root.glob("*.revalidated_*.metrics.json"))), 1)
                write_json(sidecar, {"settings": settings | {"max_new_tokens": 4096}, "generation_key": digest(settings)})
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    infer(data, trained, output, root, 42, 8192, validate_only=True)
            # LAIC inputs differ from dev; reload verification must still use the saved dev probe.
            refs = [{"id": "laic-a", "split": "test", "facts": "甲实施行为。", "charge": "示例", "sentence_months": 12}]
            prompts = messages(refs[0], "direct")
            test_jobs = [{"id": "laic-a", "split": "test", "variant": "direct", "messages": prompts,
                          "prompt_hash": digest(prompts)}]
            write_rows(data / "test0.references.jsonl", refs)
            write_rows(data / "direct.test0.jobs.jsonl", test_jobs)
            write_json(data / "manifest.json", {"tests": [{"name": "test0", "n": 1, "hash": digest(refs)}]})
            write_json(root / "token_budget.json", {"training_hash": "synthetic", "dev_jobs_hash": digest(jobs), "max_length": 8192})
            test_settings = settings | {"jobs_hash": digest(test_jobs), "split": "test0", "max_model_len": 32768}
            test_output = root / "direct.test0.predictions.jsonl"
            write_rows(test_output, [{"id": "laic-a", "prompt_hash": digest(prompts), "generation_key": digest(test_settings)}])
            write_json(str(test_output) + ".manifest.json", {"settings": test_settings, "generation_key": digest(test_settings)})
            with patch("egc.tera_server.require_server"), patch("egc.tera_server.tokenizer_for", return_value=(None, {})), \
                 patch("egc.tera_server.prompt_features", side_effect=lambda tok, msgs: (ids[::-1] if msgs else ids, self.meta, [])), \
                 patch("egc.tera_server.load_model", return_value=(self.model, 1, "lora")):
                infer(data, trained, test_output, root, 42, 32768, validate_only=True, split="test0")

    def test_post_training_failure_preserves_weights_and_incomplete_status(self):
        from types import SimpleNamespace
        from egc.tera_server import finish_training, file_hash
        tokenizer = SimpleNamespace(save_pretrained=lambda path: write_json(Path(path) / "synthetic.json", {}))
        for fail in ("threshold", "exception"):
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                def probe(*args):
                    saved = read_json(out / "completion.json")
                    self.assertTrue(saved["training_complete"])
                    self.assertFalse(saved["complete"])
                    self.assertTrue((out / "artifact/memory.pt").is_file())
                    if fail == "exception":
                        raise RuntimeError("synthetic diagnostic failure")
                    return {"passed": False, "cached_vs_full_max_abs": 0.3125}
                with patch("egc.tera_server.validate_cache", side_effect=probe), self.assertRaises((ValueError, RuntimeError)):
                    finish_training(self.model, tokenizer, out, {"arm": "tera"}, {"global_step": 894}, [], {})
                done = read_json(out / "completion.json")
                self.assertFalse(done["complete"])
                self.assertEqual(done["validation_status"], "failed")
                self.assertEqual(done["global_step"], 894)
                self.assertTrue(all(file_hash(out / "artifact" / name) == expected for name, expected in done["artifact_sha256"].items()))

    def test_generic_does_not_route_by_aux_labels_and_parameter_budget(self):
        from egc.tera_model import EvidenceAdapter
        generic = EvidenceAdapter(32, rank=8, generic=True)
        torch.nn.init.normal_(generic.up.weight)
        hidden = torch.randn(11, 32)
        memory = generic.memory(hidden, self.meta)
        before = generic(hidden[7:], memory)
        changed = dict(memory, roles=memory["roles"] + torch.randn_like(memory["roles"]) * 100)
        torch.testing.assert_close(before, generic(hidden[7:], changed), atol=0, rtol=0)
        sizes = [sum(p.numel() for p in EvidenceAdapter(3584, generic=g).parameters()) for g in (False, True)]
        self.assertLess(sizes[0], 3_000_000)
        self.assertLess(abs(sizes[1] / sizes[0] - 1), 0.05)

    def test_cpu_trainer_contract_and_accumulation(self):
        from transformers import Trainer, TrainingArguments
        from egc.tera_server import collate
        row = {"input_ids": self.ids[0].tolist(), "labels": [-100]*8 + [13, 14, 15],
               "meta": self.meta, "role_labels": [0, 1]}
        with tempfile.TemporaryDirectory() as temp:
            args = TrainingArguments(output_dir=temp, use_cpu=True, max_steps=2, per_device_train_batch_size=1,
                gradient_accumulation_steps=2, report_to=[], save_strategy="no", logging_strategy="no",
                remove_unused_columns=False, dataloader_pin_memory=False, disable_tqdm=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
            trainer = Trainer(model=self.model, args=args, train_dataset=[row]*4, data_collator=collate)
            trainer.model_accepts_loss_kwargs = False
            trainer.train()
            self.assertEqual(trainer.state.global_step, 2)
            self.assertTrue(torch.isfinite(self.model.adapter.up.weight).all())
            self.assertGreater(self.model.adapter.up.weight.abs().sum().item(), 0)

    def test_checkpoint_options_forwarded_without_dropping_or_swallowing(self):
        received = []
        def modern_api(gradient_checkpointing_kwargs=None, every_n_layers=1, offload=False):
            received.append((gradient_checkpointing_kwargs, every_n_layers, offload))
            return "enabled"
        for arm in ("direct", "generic", "tera_noaux", "tera"):
            from egc.tera_model import MemoryCausalLM
            model = MemoryCausalLM(self.model.base, arm, rank=8)
            with patch.object(model.backbone, "gradient_checkpointing_enable", side_effect=modern_api):
                self.assertEqual(model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}, every_n_layers=2, offload=True), "enabled")
                self.assertEqual(received[-1], ({"use_reentrant": False}, 2, True))
                with self.assertRaises(TypeError):
                    model.gradient_checkpointing_enable(unknown_option=True)
        # Old callers supply only the checkpoint kwargs; defaults are left to the backbone.
        with patch.object(self.model.backbone, "gradient_checkpointing_enable") as old_api:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            old_api.assert_called_once_with(gradient_checkpointing_kwargs={"use_reentrant": False})

    def test_real_peft_gradient_reload_and_interleaved_requests(self):
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError:
            self.skipTest("PEFT not installed; run this check in the server environment")
        from egc.tera_model import MemoryCausalLM
        base = self.model.base
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base.save_pretrained(root / "base")
            backbone = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
            model = MemoryCausalLM(backbone, "tera", rank=8)
            model.gradient_checkpointing_enable({"use_reentrant": False})
            model.train()
            loss = model(input_ids=self.ids, labels=torch.tensor([[-100]*8 + [13, 14, 15]]),
                         meta=self.meta, role_labels=[0, 1])["loss"]
            loss.backward()
            self.assertTrue(all(not p.requires_grad for n, p in backbone.named_parameters() if "lora_" not in n))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in backbone.named_parameters() if "lora_" in n))
            torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
            model.eval()
            backbone.save_pretrained(root / "lora")
            restored = MemoryCausalLM(PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(root / "base"), root / "lora"), "tera", rank=8)
            restored.adapter.load_state_dict(model.adapter.state_dict())
            restored.eval()
            with torch.no_grad():
                a, mem_a, past_a = model.step(self.ids[:, :8], meta=self.meta)
                b, _, _ = restored.step(self.ids[:, :8], meta=self.meta)
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                second_ids = self.ids[:, :8].clone()
                second_ids[0, 1] = 44
                _, mem_b, past_b = model.step(second_ids, meta=self.meta)
                next_a, _, _ = model.step(self.ids[:, 8:9], memory=mem_a, past=past_a)
                next_b, _, _ = model.step(self.ids[:, 8:9], memory=mem_b, past=past_b)
                for ids, observed in ((self.ids[:, :8], next_a), (second_ids, next_b)):
                    _, mem, past = model.step(ids, meta=self.meta)
                    expected, _, _ = model.step(self.ids[:, 8:9], memory=mem, past=past)
                    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
