"""Exercise both training dispatch/export paths with CPU mocks; no model/API calls."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from egc.cli import parser
from egc.io import read_json, write_json, write_rows
from egc.prepare import prepare
from egc.server import select_training_mode, tokenize_sft_row, train
from test_pipeline import row


def tokenizer_fixture():
    """Small deterministic renderer; exercises the flag contract without downloading a model."""
    tokenizer = Mock(pad_token_id=0, chat_template="template using enable_thinking")
    def render(messages, tokenize, add_generation_prompt, enable_thinking):
        if tokenize or enable_thinking:
            raise AssertionError("Must explicitly render non-thinking text")
        prefix = "<user>" + messages[0]["content"] + "<assistant><think>\n\n</think>\n\n"
        return prefix if add_generation_prompt else prefix + messages[-1]["content"] + "<eos>"
    tokenizer.apply_chat_template.side_effect = render
    tokenizer.encode.side_effect = lambda text, add_special_tokens: list(map(ord, text))
    return tokenizer


class TrainingTests(unittest.TestCase):
    def test_explicit_rendering_and_completion_mask(self):
        tokenizer = tokenizer_fixture()
        _, rows, _ = prepare([row()], "base")
        result = tokenize_sft_row(tokenizer, rows[0])
        prompt = tokenizer.apply_chat_template.call_args_list[0]
        self.assertFalse(prompt.kwargs["enable_thinking"])
        self.assertTrue(prompt.kwargs["add_generation_prompt"])
        self.assertEqual(len(result["labels"]), len(result["input_ids"]))
        first_label = result["completion_mask"].index(1)
        self.assertEqual(result["labels"][:first_label], [-100] * first_label)
        self.assertEqual(result["labels"][first_label:], result["input_ids"][first_label:])
        supervised = "".join(map(chr, result["labels"][first_label:]))
        self.assertEqual(supervised, rows[0]["completion"][0]["content"] + "<eos>")
        self.assertNotIn("<think>", supervised)
        for call in tokenizer.encode.call_args_list:
            self.assertFalse(call.kwargs["add_special_tokens"])

    def test_mismatched_template_or_token_boundary_rejected(self):
        _, rows, _ = prepare([row()], "base")
        tokenizer = tokenizer_fixture()
        tokenizer.apply_chat_template.side_effect = ["prefix", "different-prefix-answer"]
        with self.assertRaisesRegex(ValueError, "inference prefix"):
            tokenize_sft_row(tokenizer, rows[0])
        tokenizer = tokenizer_fixture()
        tokenizer.encode.side_effect = [[1, 2], [1, 3, 4]]
        with self.assertRaisesRegex(ValueError, "token boundary"):
            tokenize_sft_row(tokenizer, rows[0])
        tokenizer = tokenizer_fixture()
        tokenizer.encode.side_effect = [[1, 2], [1, 2]]
        with self.assertRaisesRegex(ValueError, "token boundary"):
            tokenize_sft_row(tokenizer, rows[0])

    def test_parameter_boundary_and_explicit_policy(self):
        for count, mode in ((1_700_000_000, "full"), (4_000_000_000, "full"),
                            (6_999_999_999, "full"), (7_000_000_000, "lora"), (8_000_000_000, "lora")):
            self.assertEqual(select_training_mode(count), mode)
            self.assertEqual(select_training_mode(count, mode), mode)
            with self.assertRaises(ValueError):
                select_training_mode(count, "lora" if mode == "full" else "full")
        for invalid in (0, -1, True, 4.0):
            with self.assertRaises(ValueError):
                select_training_mode(invalid)

    def test_full_and_lora_export_with_mock_trainer(self):
        for mode, count, folder in (("full", 4_000_000_000, "model"), ("lora", 8_000_000_000, "adapter")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_dir, out = root / "base", root / "run"
                write_json(model_dir / "config.json", {"max_position_embeddings": 4096})
                _, train_rows, _ = prepare([row(number=1)], "base")
                _, dev_rows, _ = prepare([row("dev", number=2, facts="独立验证案情")], "base")
                write_rows(root / "train.jsonl", train_rows)
                write_rows(root / "dev.jsonl", dev_rows)
                args = parser().parse_args(["train", "--model", str(model_dir), "--training-mode", mode,
                         "--train", str(root / "train.jsonl"), "--dev", str(root / "dev.jsonl"), "--output", str(out)])
                model = Mock()
                model.num_parameters.side_effect = lambda only_trainable=False: count if mode == "full" or not only_trainable else 1234
                tokenizer = tokenizer_fixture()
                captured = {}

                class Config:
                    # Reproduces the user's SFTConfig without chat_template_kwargs.
                    def __init__(self, completion_only_loss=None, **kwargs):
                        if "chat_template_kwargs" in kwargs:
                            raise AssertionError("Unsupported SFTConfig argument")

                class Trainer:
                    def __init__(self, processing_class=None, **kwargs):
                        captured.update(kwargs)
                        self.model = kwargs["model"]
                        self.state = SimpleNamespace(global_step=4, best_model_checkpoint="dev-best")

                    def is_world_process_zero(self):
                        return True

                    def train(self, **kwargs):
                        pass

                    def save_model(self, path):
                        captured["export"] = path

                modules = {
                    "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
                                              bfloat16="bf16", float16="fp16"),
                    "transformers": SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
                                                     AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)),
                    "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda value: value)),
                    "trl": SimpleNamespace(SFTConfig=Config, SFTTrainer=Trainer),
                    # Full SFT must succeed even when PEFT cannot be imported.
                    "peft": None if mode == "full" else SimpleNamespace(LoraConfig=lambda **kwargs: kwargs),
                }
                with patch.dict("sys.modules", modules), patch("egc.server.environment", return_value={}):
                    train(args)
                self.assertEqual(captured["export"], str(out / folder))
                self.assertEqual("peft_config" in captured, mode == "lora")
                for sample in captured["train_dataset"] + captured["eval_dataset"]:
                    self.assertEqual(set(sample), {"input_ids", "attention_mask", "labels", "completion_mask"})
                    self.assertIn(-100, sample["labels"])
                    self.assertTrue(any(label != -100 for label in sample["labels"]))
                if mode == "full":
                    model.float.assert_called_once_with()
                    model.requires_grad_.assert_called_once_with(True)
                else:
                    model.float.assert_not_called()
                    model.requires_grad_.assert_not_called()
                tokenizer.save_pretrained.assert_called_once_with(str(out / folder))
                self.assertEqual(read_json(out / "completion.json")["training_mode"], mode)
                manifest = read_json(out / "run_manifest.json")
                self.assertEqual(manifest["total_parameters"], count)
                self.assertEqual(manifest["trainable_parameters"], count if mode == "full" else 1234)


if __name__ == "__main__":
    unittest.main()
