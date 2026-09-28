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
    def test_gate_fresh_full_runs_and_failure_collection(self):
        from egc.tera_server import ARMS, run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "input.zip"
            archive.write_bytes(b"synthetic")
            commands = []
            def execute(command, log):
                commands.append(command)
                out = Path(command[command.index("--output") + 1])
                if command[3] == "infer":
                    write_rows(out, [{"id": str(i), "text": '{"reasoning":"ok","sentence_months":12}', "finish_reason": "stop"} for i in range(12)])
                return {"exit_code": 0}
            with patch("egc.tera_server.require_server"), patch("egc.tera_server.unpack"), patch("egc.tera_server.prepare"), \
                 patch("egc.tera_server.execute", side_effect=execute), patch("egc.tera_server.evaluate") as score, \
                 patch("egc.tera_server.collect", return_value="synthetic_results.zip"), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
                run(archive, root / "run", model=root)
                self.assertEqual(len(commands), 10)
                self.assertIn("32", commands[0])
                self.assertEqual([c[c.index("--arm")+1] for c in commands[2::2]], list(ARMS))
                self.assertTrue(all("-1" in c for c in commands[2::2]))
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
