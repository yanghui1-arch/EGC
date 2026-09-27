"""CPU weight inspection, or --generate for a two-case server GPU probe; no training."""
import argparse
import json
import math
from pathlib import Path

from .io import digest, index_unique, read_json, read_rows, write_json


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


def json_prefix(text):
    """Locate a completed object for next-token diagnostics, never for scoring."""
    text = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    return text[:end] if isinstance(value, dict) else None


def probe(run_dir):
    import os
    import platform
    if platform.system() != "Linux":
        raise ValueError("Generation must run on the user's Linux GPU server")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from .evaluate import parse_output
    from .learned_server import check_prediction_identity
    from .server import configure_chat, environment, render_chat
    if not torch.cuda.is_available():
        raise ValueError("A server GPU is required for --generate")
    root = Path(run_dir)
    execution = read_json(root / "execution.json")
    completion = read_json(root / "direct/completion.json")
    if not completion.get("complete") or completion.get("training_mode") != "lora":
        raise ValueError("Probe requires the completed direct LoRA adapter")
    jobs = read_rows(root / "data/direct.dev.jobs.jsonl")
    predictions = read_rows(root / "direct.dev.predictions.jsonl")
    sidecar = read_json(root / "direct.dev.predictions.jsonl.manifest.json")
    check_prediction_identity(jobs, predictions, sidecar)
    settings = sidecar["settings"]
    model_path, adapter = execution["model"], completion["artifact"]
    if settings["model"] != model_path or settings["adapter"] != adapter or settings["temperature"] != 0:
        raise ValueError("Probe requires the original greedy direct inference settings")
    config = read_json(Path(model_path) / "config.json")
    if config != settings["model_config"] or read_json(Path(adapter) / "adapter_config.json") != settings["adapter_config"]:
        raise ValueError("Model/adapter config changed since the original run")
    inventory = [{"file": str(p.relative_to(directory)), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for directory in (Path(model_path), Path(adapter)) for p in sorted(directory.glob("*.safetensors"))]
    if inventory != settings["weight_inventory"]:
        raise ValueError("Weight file inventory changed since the original inference; preserve the original weights")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    protocol = configure_chat(tokenizer, config)
    prompts = [render_chat(tokenizer, job["messages"]) for job in jobs]
    if protocol != settings["chat_protocol"] or digest(prompts) != settings["rendered_prompts_hash"]:
        raise ValueError("Prompt serialization changed since the original run")
    torch.manual_seed(execution["seed"])
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                              torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True,
                                     autocast_adapter_dtype=False).to(dtype=torch.bfloat16).eval()
    eos = model.generation_config.eos_token_id
    stops = sorted(set((eos if isinstance(eos, list) else [eos]) + protocol["stop_token_ids"]) - {None})
    generation = GenerationConfig(max_new_tokens=settings["max_new_tokens"], do_sample=False,
                                  eos_token_id=stops, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                  use_cache=True)
    report = {"kind": "two_case_transformers_diagnostic", "run_git_commit": execution["git_commit"],
              "model": model_path, "adapter": adapter, "selection": "first two original direct dev jobs; no label selection",
              "generation_config": generation.to_dict(), "chat_protocol": protocol,
              "environment": environment(model_path, gpu=True), "cases": [],
              "note": "Diagnostic only, no MAE, no original output replacement. Engine numerics and batch size differ."}
    original = index_unique(predictions)
    with torch.inference_mode():
        for job, prompt in list(zip(jobs, prompts))[:2]:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            generated = model.generate(**inputs, generation_config=generation)
            ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            stopped = bool(ids) and ids[-1] in stops
            case = {"id": job["id"], "prompt_hash": job["prompt_hash"], "vllm_original": original[job["id"]],
                    "transformers": {"text": text, "token_ids": ids,
                                     "finish_reason": "stop" if stopped else "length",
                                     "valid": stopped and parse_output(text) is not None}}
            # Inspect the next-token distribution at the ORIGINAL answer's JSON boundary.
            # No gold labels are read, and this prefix is never used as a repaired answer.
            prefix = json_prefix(original[job["id"]]["text"])
            if prefix is not None:
                boundary = tokenizer(prompt + prefix, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                logits = model(**boundary, use_cache=False).logits[0, -1].float()
                watched = sorted(set(stops + tokenizer.encode("𬭤", add_special_tokens=False)
                                     + tokenizer.encode("䏡", add_special_tokens=False)))
                case["original_boundary_logits"] = {
                    "top10": [{"id": i, "token": tokenizer.decode([i]), "logit": float(logits[i])}
                              for i in logits.topk(10).indices.tolist()],
                    "watched": [{"id": i, "token": tokenizer.decode([i]), "logit": float(logits[i]),
                                 "rank": int((logits > logits[i]).sum()) + 1} for i in watched]}
            report["cases"].append(case)
            print(f"Completed Transformers probe {len(report['cases'])}/2", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generate", action="store_true", help="Server GPU: reuse direct adapter for two Transformers generations")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Use a new diagnostic output file; preserve previous evidence")
    report = probe(args.run_dir) if args.generate else diagnose(args.run_dir)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
