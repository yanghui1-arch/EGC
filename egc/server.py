"""Import heavy ML dependencies only when the user runs server commands."""
from __future__ import annotations

import importlib.metadata
import inspect
import os
import platform
from pathlib import Path

from .io import digest, index_unique, read_json, read_rows, write_json, write_rows

MODEL_ROOT = "/mnt/yanghui/models/Qwen"


def environment(root=MODEL_ROOT, gpu=False):
    versions = {}
    for package in ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "tensorboard", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    path = Path(root)
    configs = sorted(path.rglob("config.json")) if path.is_dir() else []
    models = []
    for config in configs:
        obj = read_json(config)
        models.append({"path": str(config.parent), "model_type": obj.get("model_type"),
                       "architectures": obj.get("architectures"), "max_position_embeddings": obj.get("max_position_embeddings")})
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "model_root": str(path), "models": models}
    if gpu:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                           "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)}
                          for i in range(torch.cuda.device_count())]
    return result


def resolve_model(path):
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError("--model must be an exact local model directory containing config.json; run doctor first")
    return str(directory)


def render_chat(tokenizer, messages):
    # Disable optional Qwen3 thinking without imposing it on Qwen2.5 templates.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def check_training_rows(train_rows, dev_rows):
    for expected, rows in (("train", train_rows), ("dev", dev_rows)):
        index_unique(rows)
        if not rows or any(r.get("split") != expected for r in rows):
            raise ValueError(f"SFT requires nonempty {expected} data; never use test as validation")
        for row in rows:
            if not row.get("prompt") or not row.get("completion"):
                raise ValueError("Missing conversational prompt/completion")
    if {r["id"] for r in train_rows} & {r["id"] for r in dev_rows}:
        raise ValueError("Train/dev case ID overlap")
    if {(r["dataset"], r["source_id"]) for r in train_rows} & {(r["dataset"], r["source_id"]) for r in dev_rows}:
        raise ValueError("Train/dev same-source case overlap")
    if len({r["variant"] for r in train_rows + dev_rows}) != 1:
        raise ValueError("Use the same variant for train and dev")
    def facts(row):
        import json
        from .data import normalized_fact
        return normalized_fact(json.loads(row["prompt"][-1]["content"])["facts"])
    if {facts(r) for r in train_rows} & {facts(r) for r in dev_rows}:
        raise ValueError("Train/dev exact fact overlap")


def train(args):
    from datasets import Dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise ValueError("Training requires the server CUDA environment")
    model_path = resolve_model(args.model)
    train_rows, dev_rows = read_rows(args.train), read_rows(args.dev)
    check_training_rows(train_rows, dev_rows)
    if args.max_length <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("Invalid sequence/batch configuration")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Nonempty run directory: use a new output or explicit --resume checkpoint")
    identity = digest({"model": model_path, "model_config": read_json(Path(model_path) / "config.json"),
                       "train": train_rows, "dev": dev_rows,
                       "args": {k: v for k, v in vars(args).items() if k not in {"resume", "output"}}})
    if args.resume:
        old = read_json(out / "run_manifest.json")
        if old.get("training_identity") != identity:
            raise ValueError("Resume model/dataset/training configuration differs from the original run")
        checkpoint = Path(args.resume).resolve()
        if not checkpoint.is_dir() or not checkpoint.is_relative_to(out.resolve()):
            raise ValueError("Resume checkpoint must be an existing checkpoint inside this run directory")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Fail instead of silently dropping the final month label during truncation.
    too_long = []
    for row in train_rows + dev_rows:
        tokens = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True, enable_thinking=False)
        if len(tokens) > args.max_length:
            too_long.append({"id": row["id"], "tokens": len(tokens)})
    if too_long:
        write_json(out / "overlength.json", too_long)
        raise ValueError(f"{len(too_long)} samples exceed max_length; inspect overlength.json, do not truncate targets")
    if "completion_only_loss" not in inspect.signature(SFTConfig).parameters or "processing_class" not in inspect.signature(SFTTrainer).parameters:
        raise ValueError("Installed TRL lacks required APIs; send doctor.json before changing the environment")
    template_options = {}
    if "chat_template_kwargs" in inspect.signature(SFTConfig).parameters:
        template_options["chat_template_kwargs"] = {"enable_thinking": False}
    elif "enable_thinking" in str(tokenizer.chat_template):
        raise ValueError("This thinking template needs TRL chat_template_kwargs support for consistent train/infer formatting")
    bf16 = args.precision == "bf16"
    if bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU lacks bf16 support; choose --precision fp16")
    tokenizer_config = read_json(Path(model_path) / "config.json")
    if tokenizer_config.get("max_position_embeddings", args.max_length) < args.max_length:
        raise ValueError("max_length exceeds model context configuration")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.bfloat16 if bf16 else torch.float16)
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank*2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    config = SFTConfig(output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
                       per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
                       gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
                       max_length=args.max_length, completion_only_loss=True, packing=False,
                       bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
                       gradient_checkpointing_kwargs={"use_reentrant": False},
                       eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps",
                       save_steps=args.eval_steps, save_total_limit=2, load_best_model_at_end=True,
                       metric_for_best_model="eval_loss", greater_is_better=False,
                       logging_steps=1, report_to=["tensorboard"], seed=args.seed, data_seed=args.seed,
                       ddp_find_unused_parameters=False, **template_options)
    def dataset(rows):
        return Dataset.from_list([{k: r[k] for k in ("prompt", "completion")} for r in rows])
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer,
                         train_dataset=dataset(train_rows), eval_dataset=dataset(dev_rows), peft_config=lora)
    if trainer.is_world_process_zero():
        write_json(out / "run_manifest.json", {"kind": "prompt_pilot_sft", "model": model_path,
                   "args": {k: v for k, v in vars(args).items() if k != "func"},
                   "train_hash": digest(train_rows), "dev_hash": digest(dev_rows), "training_identity": identity,
                   "environment": environment(args.model, gpu=True)})
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(str(out / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(out / "adapter"))
        write_json(out / "completion.json", {"complete": True, "global_step": trainer.state.global_step,
                    "best_checkpoint": trainer.state.best_model_checkpoint})


def infer(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = resolve_model(args.model)
    jobs = read_rows(args.jobs)
    index_unique(jobs)
    for job in jobs:
        if job["prompt_hash"] != digest(job["messages"]):
            raise ValueError("Job prompt hash mismatch")
        if any(k in job for k in ("opinion", "sentence_months", "completion", "reference")):
            raise ValueError("Inference jobs must not contain reference labels")
    if len({r["variant"] for r in jobs}) != 1 or len({r["split"] for r in jobs}) != 1:
        raise ValueError("Inference jobs must share one variant and split")
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.max_new_tokens >= args.max_model_len:
        raise ValueError("Invalid inference batch or context limits")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    prompts = [render_chat(tokenizer, job["messages"]) for job in jobs]
    for job, prompt in zip(jobs, prompts):
        if len(tokenizer.encode(prompt, add_special_tokens=False)) + args.max_new_tokens > args.max_model_len:
            raise ValueError(f"Prompt plus output budget exceeds context: {job['id']}")
    adapter = Path(args.adapter).resolve() if args.adapter else None
    adapter_config = read_json(adapter / "adapter_config.json") if adapter else None
    settings = {"model": model_path, "model_config": read_json(Path(model_path) / "config.json"),
                "adapter": str(adapter) if adapter else None, "adapter_config": adapter_config,
                "jobs_hash": digest(jobs), "seed": args.seed, "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
                "tensor_parallel": args.tensor_parallel, "batch_size": args.batch_size,
                "weight_inventory": [{"file": str(p.relative_to(directory)), "bytes": p.stat().st_size,
                                       "mtime_ns": p.stat().st_mtime_ns}
                                      for directory in ([Path(model_path)] + ([adapter] if adapter else []))
                                      for p in sorted(directory.glob("*.safetensors"))]}
    generation_key = digest(settings)
    output = Path(args.output)
    existing = index_unique(read_rows(output)) if output.exists() and output.stat().st_size else {}
    if not set(existing) <= {r["id"] for r in jobs}:
        raise ValueError("Output contains IDs from another dataset")
    for job in jobs:
        saved = existing.get(job["id"])
        if saved and (saved.get("generation_key") != generation_key or saved.get("prompt_hash") != job["prompt_hash"]):
            raise ValueError("Cannot resume results from different model/input/settings")
    pending = [(job, prompt) for job, prompt in zip(jobs, prompts) if job["id"] not in existing]
    if not pending:
        print("All inference rows already completed")
        return
    # Local paths are required; prevent accidental model downloads in workers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kwargs = {"model": model_path, "trust_remote_code": False, "tensor_parallel_size": args.tensor_parallel,
              "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_memory,
              "seed": args.seed, "enable_lora": adapter is not None}
    if adapter:
        kwargs["max_lora_rank"] = max(8, adapter_config["r"])
    engine = LLM(**kwargs)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, seed=args.seed)
    request = LoRARequest("egc", 1, str(adapter)) if adapter else None
    write_json(str(output) + ".manifest.json", {"settings": settings, "generation_key": generation_key,
                                              "environment": environment(args.model, gpu=True)})
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start+args.batch_size]
        generated = engine.generate([prompt for _, prompt in batch], sampling, lora_request=request)
        for (job, _), response in zip(batch, generated):
            answer = response.outputs[0]
            existing[job["id"]] = {"id": job["id"], "prompt_hash": job["prompt_hash"],
                                   "generation_key": generation_key, "text": answer.text,
                                   "finish_reason": answer.finish_reason,
                                   "prompt_tokens": len(response.prompt_token_ids), "output_tokens": len(answer.token_ids)}
        write_rows(output, [existing[j["id"]] for j in jobs if j["id"] in existing])
        print(f"Completed {len(existing)}/{len(jobs)}")
