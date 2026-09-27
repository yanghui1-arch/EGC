"""CPU-only checks of model selection and shared Qwen2 formatting configuration."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from egc.cli import parser
from egc.io import digest, read_json, write_json, write_rows
from egc.learned_server import DEFAULT_MODEL, run
from egc.server import configure_chat, infer, tokenize_sft_row, QWEN2_CHAT_TEMPLATE


def tokenizer(template=None):
    t = Mock(chat_template=template)
    markers = {"<|im_start|>":151644, "<|im_end|>":151645}
    t.encode.side_effect = lambda text, **kwargs: [markers[text]] if text in markers else list(map(ord,text))
    t.apply_chat_template.return_value = "<|im_start|>assistant\n"
    return t


class Qwen25Tests(unittest.TestCase):
    def test_base_native_eos_supervised_and_prompt_preserved(self):
        t = tokenizer("native template")
        t.eos_token_id = 151643
        markers = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}
        def encode(text, **kwargs):
            import re
            ids = []
            for part in re.split(r"(<\|(?:im_start|im_end|endoftext)\|>)", text):
                ids.extend([markers[part]] if part in markers else map(ord, part))
            return ids
        t.encode.side_effect = encode
        protocol = configure_chat(t, {"model_type": "qwen2", "eos_token_id": 151643})
        prefix = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"
        row = {"id": "synthetic", "prompt": [{"role": "user", "content": "question"}],
               "completion": [{"role": "assistant", "content": "answer"}]}
        t.apply_chat_template.side_effect = lambda messages, **kw: prefix if kw["add_generation_prompt"] else prefix + "answer<|im_end|>\n"
        result = tokenize_sft_row(t, row, protocol)
        self.assertEqual(result["input_ids"][:len(encode(prefix))], encode(prefix))
        self.assertEqual(result["labels"][:len(encode(prefix))], [-100] * len(encode(prefix)))
        self.assertEqual(result["labels"][len(encode(prefix)):], encode("answer<|endoftext|>"))
        unchanged = configure_chat(t, {"model_type": "qwen2", "eos_token_id": 151645})
        self.assertIsNone(unchanged["completion_end_token"])
        self.assertEqual(tokenize_sft_row(t, row, unchanged)["input_ids"], encode(prefix + "answer<|im_end|>\n"))
        t.apply_chat_template.side_effect = lambda messages, **kw: prefix if kw["add_generation_prompt"] else prefix + "answer"
        with self.assertRaisesRegex(ValueError, "assistant ChatML"):
            tokenize_sft_row(t, row, protocol)
        t.eos_token_id = 99
        with self.assertRaisesRegex(ValueError, "native EOS"):
            configure_chat(t, {"model_type": "qwen2", "eos_token_id": 151643})

    def test_default_and_native_template_are_preserved(self):
        self.assertTrue(DEFAULT_MODEL.endswith("/Qwen2.5-7B"))
        self.assertEqual(run.__defaults__[0], DEFAULT_MODEL)
        t = tokenizer("checkpoint template")
        p = configure_chat(t, {"model_type":"qwen2"})
        self.assertEqual(t.chat_template, "checkpoint template")
        self.assertEqual(p["source"], "checkpoint")
        self.assertEqual(p["stop_token_ids"], [151645])

    def test_missing_template_fallback_and_foreign_tokenizer_rejection(self):
        t = tokenizer()
        p = configure_chat(t, {"model_type":"qwen2"})
        self.assertEqual(t.chat_template, QWEN2_CHAT_TEMPLATE)
        self.assertEqual(p["source"], "egc-qwen2-text-chatml-v1")
        t = tokenizer()
        t.encode.return_value = [999]
        t.encode.side_effect = None
        with self.assertRaisesRegex(ValueError, "special tokens"):
            configure_chat(t, {"model_type":"qwen2"})
        self.assertIsNone(t.chat_template)
        with self.assertRaisesRegex(ValueError, "no chat template"):
            configure_chat(tokenizer(), {"model_type":"other"})

    def test_other_native_models_keep_original_stopping_policy(self):
        t = tokenizer("qwen3 native")
        p = configure_chat(t, {"model_type":"qwen3"})
        self.assertEqual(t.chat_template, "qwen3 native")
        self.assertEqual(p["stop_token_ids"], [])
        t.encode.assert_not_called()

    def test_inference_uses_and_fingerprints_qwen2_end_of_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "model/config.json", {"model_type":"qwen2"})
            prompt = [{"role":"user", "content":"synthetic"}]
            jobs = [{"id":"synthetic", "variant":"direct", "split":"dev", "messages":prompt,"prompt_hash":digest(prompt)}]
            write_rows(root / "jobs.jsonl", jobs)
            args = parser().parse_args(["infer", "--model",str(root / "model"),"--jobs",str(root / "jobs.jsonl"),"--output",str(root / "pred.jsonl")])
            engine = Mock()
            engine.generate.return_value = [SimpleNamespace(prompt_token_ids=[1], outputs=[
                SimpleNamespace(text='{"sentence_months":12}',finish_reason="stop",token_ids=[2])])]
            sampling = Mock(return_value="sampling")
            modules = {
                "transformers":SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:tokenizer())),
                "vllm":SimpleNamespace(LLM=Mock(return_value=engine),SamplingParams=sampling),
                "vllm.lora.request":SimpleNamespace(LoRARequest=Mock()),
            }
            with patch.dict("sys.modules",modules), patch.dict("os.environ",{}), patch("egc.server.environment",return_value={}):
                infer(args)
            self.assertEqual(sampling.call_args.kwargs["stop_token_ids"],[151645])
            settings = read_json(root / "pred.jsonl.manifest.json")["settings"]
            self.assertEqual(settings["chat_protocol"]["stop_token_ids"],[151645])
            self.assertEqual(settings["chat_protocol"]["source"],"egc-qwen2-text-chatml-v1")
