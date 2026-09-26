"""Read a few local weight rows on CPU; no training, generation or model writes."""
import argparse
import json
import math
from pathlib import Path

from .io import digest, read_json, write_json


def compare_rows(rows, end_id):
    reference = rows[end_id]
    if not reference or any(len(row) != len(reference) or not all(map(math.isfinite, row))
                            for row in rows.values()):
        raise ValueError("Weight rows must be finite and equally sized")
    return {str(token): {"exactly_equals_im_end": row == reference,
                        "max_absolute_difference": max(abs(a-b) for a,b in zip(row, reference))}
            for token, row in rows.items()}


def diagnose(run_dir):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    root = Path(run_dir)
    execution = read_json(root / "execution.json")
    model = Path(execution["model"])
    config = read_json(model / "config.json")
    if config.get("model_type") != "qwen2":
        raise ValueError("This diagnostic is for the returned Qwen2 run")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    # The two rare characters below were observed immediately after completed JSON.
    tokens = {text: tokenizer.encode(text, add_special_tokens=False)
              for text in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "𬭤", "䏡")}
    if tokens["<|im_end|>"] != [151645]:
        raise ValueError("Unexpected end-of-turn token mapping")
    ids = sorted({token for values in tokens.values() for token in values})
    report = {"model": str(model), "run_git_commit": execution["git_commit"],
              "tokens": tokens, "model_eos_token_id": config.get("eos_token_id"),
              "tokenizer_eos_token_id": tokenizer.eos_token_id,
              "base_vocab_hash": digest(tokenizer.get_vocab()), "adapters": {}, "weights": {}}
    for arm in ("direct", "flat", "bound"):
        adapter = Path(read_json(root / arm / "completion.json")["artifact"])
        saved = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
        report["adapters"][arm] = {
            "vocab_matches_base": saved.get_vocab() == tokenizer.get_vocab(),
            "template_matches_base": saved.chat_template == tokenizer.chat_template,
            "eos_token_id": saved.eos_token_id,
            "config": read_json(adapter / "adapter_config.json")}
    index = model / "model.safetensors.index.json"
    weight_map = read_json(index)["weight_map"] if index.exists() else {}
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        path = model / weight_map.get(name, "model.safetensors")
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            matrix = weights.get_slice(name)
            rows = {token: matrix[token:token+1].float().tolist()[0] for token in ids}
            report["weights"][name] = {"shape": matrix.get_shape(),
                                      "rows": compare_rows(rows, 151645)}
    report["note"] = ("Exact matching frozen lm_head rows imply identical mathematical logits for those tokens. "
                      "This checks selected token rows only, not all vocabulary rows or engine correctness.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Use a new diagnostic output file; preserve previous evidence")
    report = diagnose(args.run_dir)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
