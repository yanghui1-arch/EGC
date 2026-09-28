"""User-run, single-GPU TERA experiment. Transformers only; no API/test-set calls."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .evaluate import compare, parse_output, summarize
from .io import digest, read_json, read_rows, write_json, write_rows
from .learned import fresh
from .learned_server import DEFAULT_MODEL, check_prediction_identity, collect, execute, unpack
from .qwen_eos_smoke import select_jobs
from .server import SFT_TOKENIZATION_VERSION, configure_chat, environment, select_training_mode
from .tera_data import VERSION, prompt_features, training_rows

ARMS = ("direct", "generic", "tera_noaux", "tera")


def file_hash(path):
    with Path(path).open("rb") as handle:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def require_server():
    if platform.system() != "Linux":
        raise ValueError("Real training/inference runs only on the user's Linux server")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use exactly one visible CUDA GPU; default physical device is 1")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("TERA v1 requires bf16 support")


def tokenizer_for(model):
    from transformers import AutoTokenizer
    config = read_json(Path(model) / "config.json")
    if config.get("model_type") != "qwen2" or config.get("quantization_config"):
        raise ValueError("TERA v1 supports unquantized Qwen2/Qwen2.5 causal checkpoints only")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Exact character offsets require a fast tokenizer")
    return tokenizer, configure_chat(tokenizer, config)


def prepare(data, model, max_length):
    """Called by the user's server process; reuses private, already transferred data."""
    tokenizer, protocol = tokenizer_for(model)
    rows, report = training_rows(tokenizer, protocol, read_rows(data / "direct.train.sft.jsonl"),
                                 read_rows(data / "bound.train.sft.jsonl"), max_length)
    jobs = read_rows(data / "direct.dev.jobs.jsonl")
    for job in jobs:
        ids, _, _ = prompt_features(tokenizer, job["messages"])
        if len(ids) + 2048 > max_length:
            raise ValueError(f"Generation would exceed context budget: {job['id']}")
    # These private files are excluded from result collection and Git publication.
    write_rows(data / "tera.train.jsonl", rows)
    report.update({"chat_protocol": protocol, "tokenization_version": SFT_TOKENIZATION_VERSION,
                   "training_hash": digest(rows), "dev_jobs_hash": digest(jobs), "max_length": max_length,
                   "new_api_calls": 0, "all_arms_same_sft_tokens": True})
    print("Auxiliary alignment (not semantic accuracy):", json.dumps(report["alignment"]), flush=True)
    if any(report["alignment"].get(role, 0) == 0 for role in ("target", "other", "uncertain")):
        print("WARNING: at least one role has no aligned labels; this run cannot establish the full attribution mechanism", flush=True)
    write_json(data.parent / "token_budget.json", report)
    return report


def load_model(model, arm, artifact=None):
    import torch
    from transformers import AutoModelForCausalLM
    from .tera_model import MemoryCausalLM
    full_reload = artifact and read_json(artifact / "module.json")["training_mode"] == "full"
    backbone = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32 if full_reload else torch.bfloat16, attn_implementation="sdpa")
    total = backbone.num_parameters()
    mode = select_training_mode(total)
    if mode == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model
        if artifact:
            backbone = PeftModel.from_pretrained(backbone, str(artifact / "backbone"), is_trainable=False)
        else:
            backbone = get_peft_model(backbone, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    elif not artifact:
        backbone.float().requires_grad_(True)
    elif Path(model).resolve() != (artifact / "backbone").resolve():
        raise ValueError("Full-SFT inference must load the exported backbone")
    wrapped = MemoryCausalLM(backbone, arm)
    if artifact and wrapped.adapter is not None:
        wrapped.adapter.load_state_dict(torch.load(artifact / "memory.pt", map_location="cpu", weights_only=True), strict=True)
    return wrapped.cuda(), total, mode


def collate(rows):
    import torch
    if len(rows) != 1:
        raise ValueError("TERA v1 uses batch size one, with gradient accumulation")
    row = rows[0]
    return {"input_ids": torch.tensor([row["input_ids"]]), "labels": torch.tensor([row["labels"]]),
            "meta": row["meta"], "role_labels": row["role_labels"]}


def cache_probe(model, ids, meta, steps=1, precision="bf16"):
    """Compare identical weights/tokens with and without cache; keep the 0.25 gate."""
    from contextlib import nullcontext
    import torch
    if not 1 <= steps <= 4:
        raise ValueError("Cache diagnostic is bounded to 1-4 continuation steps")
    if precision not in ("bf16", "fp32"):
        raise ValueError("Unknown diagnostic precision")
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.tensor([ids], device=device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()
    checks = []
    with torch.inference_mode(), autocast:
        first, memory, past = model.step(tokens, meta=meta)
        next_id = first.argmax().reshape(1, 1)
        for position in range(steps):
            cached, _, past, base_cached = model.step(next_id, memory=memory, past=past, return_base_logits=True)
            tokens = torch.cat((tokens, next_id), 1)
            full = model.base.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[0]
            tail = full[-1:]
            base_full = model.base.lm_head(tail)[0].float()
            memory_error = 0.0
            if model.adapter is not None:
                fresh_memory = model.adapter.memory(full, meta)
                memory_error = max(float((fresh_memory[k].float() - memory[k].float()).abs().max()) for k in memory)
                tail = model.adapter(tail, fresh_memory)
            recomputed = model.base.lm_head(tail)[0].float()
            if not all(torch.isfinite(x).all() for x in (cached, recomputed, base_cached, base_full)):
                raise ValueError("Non-finite cache diagnostic logits")
            difference = cached - recomputed
            worst = int(difference.abs().argmax())
            error = float(difference.abs().max())
            equal = bool(cached.argmax() == recomputed.argmax())
            def top_summary(logits):
                top = logits.topk(2)
                return {"token": int(top.indices[0]), "margin": float(top.values[0]-top.values[1])}
            checks.append({"position": position + 1, "cached_vs_full_max_abs": error,
                "cached_vs_full_rms": float(difference.square().mean().sqrt()), "memory_max_abs": memory_error,
                "argmax_equal": equal, "cached_greedy": top_summary(cached), "full_greedy": top_summary(recomputed),
                "worst_token": worst, "worst_cached_logit": float(cached[worst]), "worst_full_logit": float(recomputed[worst]),
                "backbone_max_abs": float((base_cached-base_full).abs().max()),
                "backbone_argmax_equal": bool(base_cached.argmax() == base_full.argmax()),
                "module_increment_difference_max_abs": float((difference-(base_cached-base_full)).abs().max()),
                "passed": error <= 0.25 and memory_error <= 0.25 and equal})
            next_id = cached.argmax().reshape(1, 1)
        top = first.topk(8)
        return {"version": "cache-diagnostic-v2", "precision": precision if device.type == "cuda" else "cpu_float32",
                "passed": all(c["passed"] for c in checks),
                "cached_vs_full_max_abs": max(c["cached_vs_full_max_abs"] for c in checks),
                "memory_max_abs": max(c["memory_max_abs"] for c in checks),
                "argmax_equal": all(c["argmax_equal"] for c in checks), "first_token": int(first.argmax()),
                "first_top_ids": top.indices.tolist(), "first_top_logits": top.values.tolist(),
                "steps": checks, "note": "Unchanged absolute tolerance 0.25 and equal greedy token; backbone includes its trained LoRA"}


def finish_training(network, tokenizer, out, manifest, completion, ids, meta):
    """Export before post-training validation can discard an expensive run."""
    import torch
    artifact = out / "artifact"
    network.backbone.save_pretrained(artifact / "backbone", safe_serialization=True)
    tokenizer.save_pretrained(artifact / "tokenizer")
    if network.adapter is not None:
        torch.save(network.adapter.state_dict(), artifact / "memory.pt")
    write_json(artifact / "module.json", manifest)
    checksums = {p.relative_to(artifact).as_posix(): file_hash(p) for p in artifact.rglob("*") if p.is_file()}
    done = completion | {"complete": False, "training_complete": True, "validation_status": "pending",
                        "artifact": str(artifact), "artifact_sha256": checksums}
    write_json(out / "completion.json", done)
    try:
        done["cache_probe"] = validate_cache(network, ids, meta)
        if not done["cache_probe"]["passed"]:
            raise ValueError("Cached decode check failed; trained weights preserved; inspect completion.json cache_probe")
        done.update(complete=True, validation_status="passed")
    except Exception as exc:
        done.update(validation_status="failed", validation_error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(out / "completion.json", done)


def validate_cache(network, ids, meta):
    """BF16 decoding agreement plus an independent FP32 cache correctness check."""
    import torch
    report = cache_probe(network, ids, meta, steps=4)
    original_parameters = {name: tensor.dtype for name, tensor in network.named_parameters() if tensor.is_floating_point()}
    original_buffers = {name: tensor.dtype for name, tensor in network.named_buffers() if tensor.is_floating_point()}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        network.float()
        control = cache_probe(network, ids, meta, steps=4, precision="fp32")
    finally:
        # Restore each dtype separately: LoRA/module FP32 parameters must stay FP32.
        for names, getter in ((original_parameters, network.get_parameter), (original_buffers, network.get_buffer)):
            for name, dtype in names.items():
                tensor = getter(name)
                tensor.data = tensor.data.to(dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return report | {"validation_version": "bf16-greedy-fp32-cache-v1",
        "legacy_absolute_gate_passed": report["passed"], "fp32_control": control,
        "passed": report["argmax_equal"] and report["memory_max_abs"] <= 0.25
                  and control["argmax_equal"] and control["cached_vs_full_max_abs"] <= 1e-3
                  and control["memory_max_abs"] <= 1e-4,
        "note": "Four continuations: BF16 greedy agreement/memory<=0.25; FP32 logits<=0.001, memory<=0.0001, greedy agreement; TF32 disabled. BF16 absolute error is diagnostic, not the gate."}


def train(data, output, model, arm, seed, epochs, max_steps, max_length):
    require_server()
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from .tera_model import EvidenceAdapter
    data, out = Path(data), fresh(output).resolve()
    tokenizer, protocol = tokenizer_for(model)
    rows = read_rows(data / "tera.train.jsonl")
    budget = read_json(data.parent / "token_budget.json")
    if digest(rows) != budget["training_hash"] or protocol != budget["chat_protocol"] or max_length != budget["max_length"]:
        raise ValueError("Prepared data/protocol changed")
    if any(len(r["input_ids"]) > max_length for r in rows):
        raise ValueError("Overlength training case")
    set_seed(seed)
    network, total, mode = load_model(model, arm)
    if network.config.max_position_embeddings < max_length:
        raise ValueError("Requested context exceeds checkpoint configuration")
    network.base.config.use_cache = False
    sizes = {name: sum(p.numel() for p in EvidenceAdapter(network.config.hidden_size, generic=generic).parameters())
             for name, generic in (("tera", False), ("generic", True))}
    if sizes["tera"] >= 3_000_000 or abs(sizes["generic"] / sizes["tera"] - 1) > 0.05:
        raise ValueError("Module parameter budget/control match failed")
    manifest = {"version": VERSION, "arm": arm, "model": str(Path(model).resolve()), "training_mode": mode,
                "total_parameters": total, "module_parameters": sizes, "aux_weight": network.aux_weight,
                "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                "training_hash": digest(rows), "seed": seed, "epochs": epochs, "max_steps": max_steps,
                "max_length": max_length, "chat_protocol": protocol, "tokenization_version": SFT_TOKENIZATION_VERSION,
                "optimizer": {"lr": 1e-5, "weight_decay": 0.0, "scheduler": "linear", "warmup_steps": 0},
                "batch_size": 1, "grad_accum": 16, "checkpoint_selection": "final",
                "environment": environment(root=model, gpu=True),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    write_json(out / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    gradient_checks = {}

    class GradCheck(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if state.global_step > 1:
                return
            for group, params in (("backbone", network.backbone.parameters()),
                                  ("module", network.adapter.parameters() if network.adapter else [])):
                gradients = [p.grad for p in params if p.requires_grad and p.grad is not None]
                if group == "module" and network.adapter is None:
                    continue
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError(f"Missing/non-finite {group} gradients")
                norm = max(float(g.abs().max()) for g in gradients)
                if norm == 0:
                    raise ValueError(f"Zero {group} gradients")
                gradient_checks[f"step{state.global_step + 1}_{group}_max_abs"] = norm

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                logs.update({"last_microbatch_" + k: v for k, v in network.last_losses.items()})

    arguments = TrainingArguments(output_dir=str(out), num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-5,
        weight_decay=0.0, warmup_steps=0, lr_scheduler_type="linear", optim="adamw_torch",
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="no", logging_steps=1, report_to=["tensorboard"],
        seed=seed, data_seed=seed, remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=network, args=arguments, train_dataset=rows, data_collator=collate, callbacks=[GradCheck()])
    trainer.model_accepts_loss_kwargs = False  # loss averages each case; Trainer handles accumulation
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    if not gradient_checks:
        raise ValueError("Trainer did not execute gradient checks; unsupported callback lifecycle")
    network.eval()
    module_update = float(network.adapter.up.weight.detach().abs().max()) if network.adapter else None
    if module_update is not None and module_update == 0:
        raise ValueError("Module residual remained zero after training")
    probe_job = read_rows(data / "direct.dev.jobs.jsonl")[0]
    ids, meta, _ = prompt_features(tokenizer, probe_job["messages"])
    finish_training(network, tokenizer, out, manifest, {"arm": arm,
        "training_mode": mode, "global_step": trainer.state.global_step, "train_metrics": result.metrics,
        "module_up_max_abs": module_update,
        "gradient_checks": gradient_checks,
        "seconds": time.time() - started, "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3}, ids, meta)


def infer(data, trained, output, model, seed, max_length, smoke=False, validate_only=False):
    require_server()
    import torch
    from transformers import set_seed
    trained, data, output = Path(trained), Path(data), Path(output)
    if not validate_only and (output.exists() or Path(str(output) + ".manifest.json").exists()):
        raise ValueError("Refusing to overwrite predictions")
    done = read_json(trained / "completion.json")
    if not (done["complete"] or done.get("training_complete")):
        raise ValueError("Training incomplete")
    artifact = Path(done["artifact"])
    if artifact.resolve() != (trained / "artifact").resolve():
        raise ValueError("Foreign artifact path")
    actual = {p.relative_to(artifact).as_posix(): file_hash(p) for p in artifact.rglob("*") if p.is_file()}
    if actual != done["artifact_sha256"]:
        raise ValueError("Missing/changed backbone or module artifact")
    manifest = read_json(artifact / "module.json")
    if manifest["version"] != VERSION or manifest["model"] != str(Path(model).resolve()):
        raise ValueError("Wrong module version/base model")
    tokenizer, protocol = tokenizer_for(model)
    if protocol != manifest["chat_protocol"]:
        raise ValueError("Changed tokenization protocol")
    jobs = read_rows(data / "direct.dev.jobs.jsonl")
    budget = read_json(data.parent / "token_budget.json")
    if (digest(jobs) != budget["dev_jobs_hash"] or manifest["training_hash"] != budget["training_hash"]
            or manifest["seed"] != seed or manifest["max_length"] != max_length):
        raise ValueError("Changed inference experiment identity")
    if smoke:
        jobs = select_jobs(jobs)
    set_seed(seed)
    load_path = str(artifact / "backbone") if done["training_mode"] == "full" else model
    network, _, _ = load_model(load_path, manifest["arm"], artifact)
    network.eval()
    ids, meta, _ = prompt_features(tokenizer, jobs[0]["messages"])
    probe = validate_cache(network, ids, meta)
    if not probe["passed"]:
        write_json(output.with_suffix(".cache.metrics.json"), probe)
        raise ValueError("Cached decode check failed; inspect saved cache.metrics.json; weights are unchanged")
    original = done["cache_probe"]
    if probe["first_top_ids"] != original["first_top_ids"] or max(abs(a-b) for a, b in zip(probe["first_top_logits"], original["first_top_logits"])) > 0.01:
        raise ValueError("Save/reload logits changed")
    settings = {"version": VERSION, "arm": manifest["arm"], "model": str(Path(model).resolve()),
                "artifact_hash": digest(actual), "jobs_hash": digest(jobs), "training_hash": manifest["training_hash"],
                "backend": "transformers_request_local_cache", "seed": seed, "temperature": 0,
                "max_new_tokens": 2048, "max_model_len": max_length, "dtype": "bfloat16", "chat_protocol": protocol}
    key = digest(settings)
    sidecar = {"settings": settings, "generation_key": key, "save_reload_probe": probe}
    if validate_only:
        existing = read_json(str(output) + ".manifest.json")
        if existing["settings"] != settings:
            raise ValueError("Existing prediction settings differ; refusing reuse")
        check_prediction_identity(jobs, read_rows(output), existing)
        write_json(output.parent / f"{output.stem}.revalidated_{time.time_ns()}.metrics.json", sidecar)
        print("Revalidated saved weights and reused predictions:", output, flush=True)
        return
    write_json(str(output) + ".manifest.json", sidecar)
    eos = network.config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos]) | set(protocol["stop_token_ids"])
    predictions = []
    with output.open("x", encoding="utf-8") as handle, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, job in enumerate(jobs, 1):
            if job["prompt_hash"] != digest(job["messages"]) or job["split"] != "dev":
                raise ValueError("Changed prompt or unfrozen split")
            ids, meta, _ = prompt_features(tokenizer, job["messages"])
            if len(ids) + 2048 > max_length:
                raise ValueError("Overlength generation; no truncation")
            memory = past = None
            tokens = torch.tensor([ids], device="cuda")
            generated, finish = [], "length"
            routing = None
            started = time.time()
            for _ in range(2048):
                logits, memory, past = network.step(tokens, meta=meta, memory=memory, past=past)
                if routing is None and memory is not None:
                    probabilities = memory["roles"].float().softmax(-1)
                    routing = {"predicted_role_mean": probabilities.mean(0).tolist(),
                        "role_entropy_mean": float(-(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()),
                        "sentences": len(probabilities), "semantic_accuracy": None}
                token = int(logits.argmax())
                generated.append(token)
                if token in stops:
                    finish = "stop"
                    break
                tokens = torch.tensor([[token]], device="cuda")
            row = {"id": job["id"], "prompt_hash": job["prompt_hash"], "generation_key": key,
                   "text": tokenizer.decode(generated, skip_special_tokens=True), "finish_reason": finish,
                   "generated_tokens": len(generated), "prompt_tokens": len(ids), "seconds": time.time()-started}
            if routing is not None:
                row["routing_diagnostic"] = routing
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            predictions.append(row)
            del memory, past
            print(f"{manifest['arm']}: dev {index}/{len(jobs)}, tokens={len(generated)}, finish={finish}", flush=True)
    check_prediction_identity(jobs, predictions, sidecar)


def evaluate(data, root):
    refs = read_rows(data / "dev.references.jsonl")
    jobs = read_rows(data / "direct.dev.jobs.jsonl")
    predictions, metrics = {}, {}
    for arm in ARMS:
        path = root / f"{arm}.dev.predictions.jsonl"
        predictions[arm] = read_rows(path)
        check_prediction_identity(jobs, predictions[arm], read_json(str(path) + ".manifest.json"))
        metrics[arm] = summarize(refs, predictions[arm])
        metrics[arm]["predicted_months"] = dict(Counter(str(parse_output(p["text"])["sentence_months"])
            for p in predictions[arm] if p["finish_reason"] == "stop" and parse_output(p["text"]) is not None))
        metrics[arm]["by_charge"] = {charge: summarize([r for r in refs if r["charge"] == charge],
            [p for p in predictions[arm] if p["id"] in {r["id"] for r in refs if r["charge"] == charge}])
            for charge in sorted({r["charge"] for r in refs})}
    comparisons = {}
    for baseline, candidate in (("direct", "generic"), ("direct", "tera"), ("generic", "tera"), ("tera_noaux", "tera")):
        comparisons[f"{candidate}_vs_{baseline}"] = compare(refs, predictions[baseline], predictions[candidate]) if all(
            metrics[a]["eligible_for_full_mae_comparison"] for a in (baseline, candidate)) else {"eligible": False, "reason": "incomplete_valid_coverage"}
    write_json(root / "dev.metrics.json", {"version": VERSION, "metrics": metrics, "comparisons": comparisons,
        "limitations": "Single seed; weak-label semantics unverified; no original-paper reproduction; no test evaluation"})


def diagnose_cache(run_dir):
    """No training: inspect saved artifacts and three fixed dev prompts per model."""
    require_server()
    import gc
    import torch
    from transformers import set_seed
    root = Path(run_dir).resolve()
    execution = read_json(root / "execution.json")
    if execution["version"] != VERSION:
        raise ValueError("Wrong experiment version")
    budget = read_json(root / "token_budget.json")
    jobs = read_rows(root / "data/direct.dev.jobs.jsonl")
    if digest(jobs) != budget["dev_jobs_hash"] or any(j["prompt_hash"] != digest(j["messages"]) or j["split"] != "dev" for j in jobs):
        raise ValueError("Changed dev prompts; refusing diagnosis")
    tokenizer, protocol = tokenizer_for(execution["model"])
    if protocol != budget["chat_protocol"]:
        raise ValueError("Changed tokenizer protocol")
    selected = jobs[:3]  # fixed input order, no gold/reference-based selection
    path = root / f"cache_diagnostic_{time.time_ns()}.metrics.json"
    report = {"kind": "cache_diagnostic_only", "run_dir": str(root), "version": VERSION,
        "execution_hash": digest(execution), "training_hash": budget["training_hash"],
        "selected_ids": [j["id"] for j in selected], "continuation_steps": 4, "fp32_control_ids": [selected[0]["id"]],
        "training_performed": False, "benchmark_scores_computed": False, "models": {},
        "limitations": "Other saved arms cannot establish the cause in an unsaved failed arm; the 0.25 gate remains unchanged"}
    write_json(path, report)
    for name in ("smoke",) + ARMS:
        trained = root / name
        entry = {"artifact_exists": (trained / "artifact").is_dir(),
                 "completion_exists": (trained / "completion.json").is_file()}
        report["models"][name] = entry
        network = None
        try:
            if not entry["artifact_exists"] or not entry["completion_exists"]:
                entry["status"] = "no_exported_checkpoint_to_probe"
                continue
            done = read_json(trained / "completion.json")
            artifact = trained / "artifact"
            if Path(done["artifact"]).resolve() != artifact.resolve() or not (done.get("complete") or done.get("training_complete")):
                raise ValueError("Unconfirmed/foreign training artifact")
            checksums = {p.relative_to(artifact).as_posix(): file_hash(p) for p in artifact.rglob("*") if p.is_file()}
            if checksums != done["artifact_sha256"]:
                raise ValueError("Artifact checksum mismatch")
            manifest = read_json(artifact / "module.json")
            arm = "tera" if name == "smoke" else name
            expected = {"version": VERSION, "arm": arm, "model": execution["model"],
                "training_hash": budget["training_hash"], "chat_protocol": protocol,
                "seed": execution["seed"], "epochs": execution["epochs"], "max_length": execution["max_length"],
                "max_steps": 32 if name == "smoke" else -1}
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError("Checkpoint experiment identity mismatch")
            entry.update(artifact_hash=digest(checksums), global_step=done["global_step"], probes=[])
            set_seed(execution["seed"])
            load_path = str(artifact / "backbone") if done["training_mode"] == "full" else execution["model"]
            network, _, _ = load_model(load_path, arm, artifact)
            for job in selected:
                ids, meta, _ = prompt_features(tokenizer, job["messages"])
                entry["probes"].append({"id": job["id"], "prompt_hash": job["prompt_hash"],
                                        "diagnostic": cache_probe(network, ids, meta, steps=4)})
            # Same saved weights/input, higher precision control; never save converted weights.
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                network.float()
                ids, meta, _ = prompt_features(tokenizer, selected[0]["messages"])
                entry["fp32_control"] = cache_probe(network, ids, meta, steps=4, precision="fp32")
            except Exception as exc:
                entry["fp32_control_error"] = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            entry["status"] = "diagnosed"
        except Exception as exc:
            entry.update(status="error", error_type=type(exc).__name__, message=str(exc))
        finally:
            del network
            gc.collect()
            torch.cuda.empty_cache()
            write_json(path, report)
            print(name + ": " + entry["status"], flush=True)
    print("Return cache diagnostic:", path, flush=True)
    print("Return experiment evidence:", collect(root), flush=True)
    return report


def run(archive, output, model=DEFAULT_MODEL, seed=42, epochs=3, max_length=8192, smoke_only=False, resume=False):
    require_server()
    if epochs <= 0 or max_length <= 2048:
        raise ValueError("Invalid experiment budget")
    out = Path(output).resolve() if resume else fresh(output).resolve()
    model = str(Path(model).resolve())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    record = {"version": VERSION, "model": model, "seed": seed, "epochs": epochs, "max_length": max_length,
              "archive_sha256": file_hash(archive), "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "arms": ARMS,
              "smoke_only": smoke_only, "commands": [], "phase": "prepare"}
    if resume:
        from .learned_audit import checked_zip
        previous = read_json(out / "execution.json")
        for key in ("version", "model", "seed", "epochs", "max_length", "archive_sha256", "smoke_only"):
            if previous[key] != record[key]:
                raise ValueError(f"Resume experiment mismatch: {key}")
        for name, content in checked_zip(archive).items():
            if (out / "data" / name).read_bytes() != content:
                raise ValueError(f"Changed original experiment data: {name}")
        budget = read_json(out / "token_budget.json")
        if digest(read_rows(out / "data/tera.train.jsonl")) != budget["training_hash"]:
            raise ValueError("Changed prepared training data")
        write_json(out / f"execution_before_resume_{time.time_ns()}.metrics.json", previous)
        record["resumed_from_execution_hash"] = digest(previous)
        if (out / "failure.json").exists():
            (out / "failure.json").rename(out / f"failure_before_resume_{time.time_ns()}.metrics.json")
    write_json(out / "execution.json", record)
    def launch(command, name):
        record["phase"] = name
        write_json(out / "execution.json", record)
        suffix = f"_{time.time_ns()}" if resume else ""
        record["commands"].append(execute(command, out / f"logs/{name}{suffix}.log"))
        write_json(out / "execution.json", record)
    try:
        if not resume:
            unpack(archive, out / "data")
            prepare(out / "data", model, max_length)
        common = ["--data", str(out / "data"), "--model", model, "--seed", str(seed), "--max-length", str(max_length)]
        def pair(arm, smoke=False):
            name = "smoke" if smoke else arm
            trained = out / name
            prediction = out / f"{name}.dev.predictions.jsonl"
            train_cmd = [sys.executable, "-m", "egc.tera_server", "train"] + common + ["--output", str(out / name),
                        "--arm", arm, "--epochs", str(epochs), "--max-steps", "32" if smoke else "-1"]
            saved = resume and (trained / "completion.json").exists()
            if resume and prediction.exists() and not saved:
                raise ValueError("Predictions exist without their saved training artifact")
            if resume and trained.exists():
                manifest = read_json(trained / "run_manifest.json")
                expected = {"version": VERSION, "arm": arm, "model": model, "seed": seed,
                    "epochs": epochs, "max_steps": 32 if smoke else -1, "max_length": max_length,
                    "training_hash": budget["training_hash"], "chat_protocol": budget["chat_protocol"],
                    "tokenization_version": SFT_TOKENIZATION_VERSION}
                if any(manifest.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Cannot reuse mismatched training: {name}")
                if not saved:
                    if (trained / "artifact").exists():
                        raise ValueError(f"Incomplete export in {name}; preserve and inspect before retrying")
                    # Keep failed logs/TensorBoard/manifest; never delete an unfinished run.
                    trained.rename(out / f"failed_{name}_{time.time_ns()}")
            if not saved:
                launch(train_cmd, "train_" + name)
            infer_cmd = [sys.executable, "-m", "egc.tera_server", "infer"] + common + ["--trained", str(out / name),
                        "--output", str(out / f"{name}.dev.predictions.jsonl")]
            reuse = resume and prediction.exists()
            launch(infer_cmd + (["--smoke"] if smoke else []) + (["--validate-only"] if reuse else []),
                   ("revalidate_" if reuse else "infer_") + name)
        pair("tera", smoke=True)
        predictions = read_rows(out / "smoke.dev.predictions.jsonl")
        valid = sum(p["finish_reason"] == "stop" and parse_output(p["text"]) is not None for p in predictions)
        gate = {"n": len(predictions), "valid": valid, "passed": len(predictions) == 12 and valid == 12,
                "note": "32 steps / 12 dev cases: termination check only, no benchmark score"}
        write_json(out / "smoke.metrics.json", gate)
        if not gate["passed"]:
            raise ValueError("Smoke termination gate failed; full runs not started")
        if not smoke_only:
            for arm in ARMS:
                pair(arm)  # Fresh base + fresh module; never resume the smoke weights.
            evaluate(out / "data", out)
        record["phase"] = "complete_smoke" if smoke_only else "complete_dev"
        write_json(out / "completion.json", {"complete": True, "phase": record["phase"], "test_evaluated": False})
    except Exception as exc:
        write_json(out / "failure.json", {"phase": record["phase"], "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(out / "execution.json", record)
        print("Return experiment evidence:", collect(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("diagnose-cache")
    diagnostic.add_argument("--run-dir", required=True)
    for name in ("run", "train", "infer"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", default=DEFAULT_MODEL)
        sub.add_argument("--output", required=True)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--max-length", type=int, default=8192)
        if name in ("run", "train"):
            sub.add_argument("--epochs", type=int, default=3)
        if name == "run":
            sub.add_argument("--archive", required=True)
            sub.add_argument("--smoke-only", action="store_true")
            sub.add_argument("--resume", action="store_true")
        else:
            sub.add_argument("--data", required=True)
        if name == "train":
            sub.add_argument("--arm", choices=ARMS, required=True)
            sub.add_argument("--max-steps", type=int, default=-1)
        if name == "infer":
            sub.add_argument("--trained", required=True)
            sub.add_argument("--smoke", action="store_true")
            sub.add_argument("--validate-only", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"run": run, "train": train, "infer": infer, "diagnose-cache": diagnose_cache}[command](**args)


if __name__ == "__main__":
    main()
