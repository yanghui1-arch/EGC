"""Server-only fixed-budget experiments; each GPU process exits before the next.

No model imports until run/preflight. Evaluate/collect work offline on CPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import zipfile

from .b0 import encoded, pack_files
from .evaluate import compare, parse_output, summarize
from .io import digest, index_unique, read_json, read_rows, write_json, write_rows
from .learned import ARMS, VERSION, fresh, messages


def unpack(archive, output):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or any("/" in n or "\\" in n or n.startswith(".") for n in names):
            raise ValueError("Unsafe/duplicate archive member")
        if sum(i.file_size for i in bundle.infolist()) > 512_000_000:
            raise ValueError("Oversized experiment archive")
        manifest = json.loads(bundle.read("manifest.json"))
        if manifest["version"] != VERSION or manifest["arms"] != list(ARMS):
            raise ValueError("Experiment protocol mismatch")
        tests = manifest["tests"]
        if [t["name"] for t in tests] != [f"test{i}" for i in range(len(tests))]:
            raise ValueError("Unexpected benchmark names")
        allowed = {"manifest.json", "train.references.jsonl", "dev.references.jsonl"}
        allowed |= {f"{a}.{s}.sft.jsonl" for a in ARMS for s in ("train", "dev")}
        allowed |= {f"{a}.dev.jobs.jsonl" for a in ARMS}
        allowed |= {f"{t['name']}.references.jsonl" for t in tests}
        allowed |= {f"{a}.{t['name']}.jobs.jsonl" for a in ARMS for t in tests}
        if set(names) != allowed | {"checksums.json"}:
            raise ValueError("Unexpected package members")
        data = {n: bundle.read(n) for n in allowed}
        if json.loads(bundle.read("checksums.json")) != {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}:
            raise ValueError("Package checksum mismatch")
    out = fresh(output)
    for name, value in data.items():
        (out / name).write_bytes(value)
    validate_data(out)
    return manifest


def validate_data(data):
    from .server import check_training_rows
    data = Path(data)
    manifest = read_json(data / "manifest.json")
    refs = {s: read_rows(data / f"{s}.references.jsonl") for s in ("train", "dev")}
    for arm in ARMS:
        sft = {s: read_rows(data / f"{arm}.{s}.sft.jsonl") for s in refs}
        check_training_rows(sft["train"], sft["dev"])
        for split in refs:
            if [r["id"] for r in sft[split]] != [r["id"] for r in refs[split]]:
                raise ValueError("Paired training case mismatch")
            for sample, reference in zip(sft[split], refs[split]):
                answer = json.loads(sample["completion"][0]["content"])
                if sample["prompt"] != messages(reference, arm) or answer["sentence_months"] != reference["sentence_months"]:
                    raise ValueError("Prompt/gold label mismatch")
        for split in ["dev"] + [t["name"] for t in manifest["tests"]]:
            reference = read_rows(data / f"{split}.references.jsonl")
            jobs = read_rows(data / f"{arm}.{split}.jobs.jsonl")
            if [r["id"] for r in jobs] != [r["id"] for r in reference]:
                raise ValueError("Inference case mismatch")
            for job, row in zip(jobs, reference):
                if job["messages"] != messages(row, arm) or job["prompt_hash"] != digest(job["messages"]):
                    raise ValueError("Inference input mismatch")
    return manifest


def preflight(data, model, max_length, output):
    from transformers import AutoTokenizer
    from .server import render_chat, tokenize_sft_row
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("This checkpoint has no chat template; use a verified Qwen3 chat checkpoint for this protocol")
    report, too_long = {}, []
    for arm in ARMS:
        report[arm] = {}
        for split in ("train", "dev"):
            rows = read_rows(Path(data) / f"{arm}.{split}.sft.jsonl")
            lengths = []
            labels = 0
            for row in rows:
                tokens = tokenize_sft_row(tokenizer, row)
                lengths.append(len(tokens["input_ids"]))
                labels += sum(tokens["completion_mask"])
                if lengths[-1] > max_length:
                    too_long.append({"id": row["id"], "arm": arm, "split": split, "tokens": lengths[-1]})
            report[arm][split] = {"cases": len(rows), "max_tokens": max(lengths), "total_tokens_per_epoch": sum(lengths),
                                  "supervised_tokens_per_epoch": labels}
        for job in read_rows(Path(data) / f"{arm}.dev.jobs.jsonl"):
            n = len(tokenizer.encode(render_chat(tokenizer, job["messages"]), add_special_tokens=False)) + 2048
            if n > max_length:
                too_long.append({"id": job["id"], "arm": arm, "split": "dev_generation", "tokens": n})
    write_json(output, {"arms": report, "overlength": too_long,
                        "note": "Equal examples/epochs, not equal supervised tokens; report this confound"})
    if too_long:
        raise ValueError("Overlength examples: inspect token_budget.json; no silent truncation or arm-specific exclusion")
    return report


def execute(command, logfile):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with logfile.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Subprocess failed (exit {code}); inspect {logfile}")
    return {"command": command, "seconds": time.time()-started, "exit_code": code}


def inference_command(model, jobs, output, max_length, seed, adapter=None):
    command = [sys.executable, "-m", "egc", "infer", "--model", str(model), "--jobs", str(jobs),
               "--output", str(output), "--batch-size", "2", "--max-model-len", str(max_length),
               "--max-new-tokens", "2048", "--tensor-parallel", "1", "--temperature", "0", "--dtype", "bfloat16", "--seed", str(seed)]
    if adapter:
        command += ["--adapter", str(adapter)]
    return command


def train_command(data, out, arm, model, seed, epochs, max_length):
    return [sys.executable, "-m", "egc", "train", "--model", str(model), "--training-mode", "auto",
            "--train", str(Path(data) / f"{arm}.train.sft.jsonl"), "--dev", str(Path(data) / f"{arm}.dev.sft.jsonl"),
            "--output", str(Path(out) / arm), "--epochs", str(epochs), "--learning-rate", "1e-5",
            "--batch-size", "1", "--grad-accum", "16", "--rank", "64", "--max-length", str(max_length),
            "--seed", str(seed), "--eval-steps", "100", "--save-total-limit", "1", "--checkpoint-selection", "final"]


def check_prediction_identity(jobs, predictions, sidecar):
    expected = index_unique(jobs)
    index_unique(predictions)
    if sidecar["settings"]["jobs_hash"] != digest(jobs):
        raise ValueError("Generation job identity mismatch")
    if sidecar["generation_key"] != digest(sidecar["settings"]):
        raise ValueError("Generation settings hash mismatch")
    if {p["id"] for p in predictions} != set(expected):
        raise ValueError("Incomplete or foreign prediction IDs")
    for pred in predictions:
        if pred["prompt_hash"] != expected[pred["id"]]["prompt_hash"] or pred["generation_key"] != sidecar["generation_key"]:
            raise ValueError("Prediction provenance mismatch")


def evidence_checks(rows, predictions, arm):
    refs = index_unique(rows)
    counts = defaultdict(int)
    for pred in predictions:
        body = parse_output(pred["text"])
        if not body or pred.get("finish_reason") == "length":
            continue
        evidence = body.get("evidence")
        if arm in {"direct", "frozen"}:
            continue
        if not isinstance(evidence, list) or not evidence:
            counts["evidence_schema_failures"] += 1
            continue
        for item in evidence:
            counts["quotes"] += 1
            quote = item.get("quote") if isinstance(item, dict) else item
            if isinstance(quote, str) and quote.strip() and quote in refs[pred["id"]]["facts"]:
                counts["source_quotes"] += 1
            if arm == "bound" and isinstance(item, dict):
                subject = item.get("subject")
                if isinstance(subject, str) and isinstance(quote, str) and subject and subject in quote:
                    counts["literal_subject_in_quote"] += 1
    return dict(counts) | {"semantic_attribution_accuracy": None,
                          "literal_subject_overlap_is_accuracy": False}


def evaluate(data, run_dir, split="dev"):
    data, root = Path(data), Path(run_dir)
    refs = read_rows(data / f"{split}.references.jsonl")
    training = read_rows(data / "train.references.jsonl")
    execution = read_json(root / "execution.json") if (root / "execution.json").exists() else None
    # Fit statistics to train only; numeric medians are statistical baselines, not charge rules.
    groups = defaultdict(list)
    for row in training:
        groups[row["charge"]].append(row["sentence_months"])
    overall = statistics.median([r["sentence_months"] for r in training])
    median_predictions = [{"id": r["id"], "finish_reason": "stop", "text": json.dumps({
        "reasoning": "train-only per-charge median", "sentence_months": round(statistics.median(groups[r["charge"]])) if groups[r["charge"]] else round(overall)})} for r in refs]
    report = {"version": VERSION, "split": split, "metrics": {"train_median": summarize(refs, median_predictions)},
              "comparisons": {}, "evidence_diagnostics": {}, "limitations": [
                  "Single-seed exploratory results; not original-paper reproduction",
                  "Character ROUGE diagnostic is not the paper metric",
                  "Literal quote checks do not establish legal reasoning or subject attribution accuracy",
                  "Equal cases/epochs but differing output token budgets used in training"]}
    predictions = {}
    for arm in ("frozen",) + ARMS:
        path = root / f"{arm}.{split}.predictions.jsonl"
        predictions[arm] = read_rows(path)
        jobs = read_rows(data / f"{'direct' if arm == 'frozen' else arm}.{split}.jobs.jsonl")
        generation = read_json(str(path) + ".manifest.json")
        check_prediction_identity(jobs, predictions[arm], generation)
        if execution:
            expected_model, expected_adapter = execution["model"], None
            if arm != "frozen":
                trained = read_json(root / arm / "completion.json")
                if trained["training_mode"] == "full":
                    expected_model = trained["artifact"]
                else:
                    expected_adapter = trained["artifact"]
            expected = {"model": str(Path(expected_model).resolve()),
                        "adapter": str(Path(expected_adapter).resolve()) if expected_adapter else None,
                        "seed": execution["seed"], "temperature": 0.0,
                        "dtype": "bfloat16",
                        "max_model_len": execution["max_length"], "max_new_tokens": 2048}
            if any(generation["settings"].get(k) != v for k, v in expected.items()):
                raise ValueError("Predictions came from a different model/adapter/generation protocol")
        report["metrics"][arm] = summarize(refs, predictions[arm])
        report["metrics"][arm]["by_charge"] = {c: summarize([r for r in refs if r["charge"] == c],
            [p for p in predictions[arm] if p["id"] in {r["id"] for r in refs if r["charge"] == c}])
            for c in sorted({r["charge"] for r in refs})}
        report["evidence_diagnostics"][arm] = evidence_checks(refs, predictions[arm], arm)
    for baseline, candidate in (("frozen", "direct"), ("direct", "flat"), ("flat", "bound")):
        key = f"{candidate}_vs_{baseline}"
        if all(report["metrics"][a]["eligible_for_full_mae_comparison"] for a in (baseline, candidate)):
            report["comparisons"][key] = compare(refs, predictions[baseline], predictions[candidate])
        else:
            report["comparisons"][key] = {"eligible": False, "reason": "incomplete_valid_coverage"}
    write_json(root / f"{split}.metrics.json", report)
    return report


def collect(run_dir):
    root = Path(run_dir)
    # Allowlist only reports, predictions and training metadata; never checkpoints or training inputs.
    files = {}
    for pattern in ("execution.json", "completion.json", "test_completion.json", "failure.json", "token_budget.json", "*.metrics.json",
                    "*.predictions.jsonl", "*.predictions.jsonl.manifest.json", "data/manifest.json",
                    "*/run_manifest.json", "*/completion.json", "logs/*.log"):
        for path in root.glob(pattern):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    if not files:
        raise ValueError("No experiment evidence to collect")
    destination = root / "results.zip"
    if destination.exists():
        destination = root / f"results_{time.time_ns()}.zip"
    pack_files(destination, files)
    return str(destination)


def run(archive, output, model="/mnt/yanghui/models/Qwen/Qwen3-1.7B", seed=42, epochs=3, max_length=8192, resume=False):
    if platform.system() != "Linux":
        raise ValueError("Training/inference must run on the user's Linux GPU server")
    if epochs <= 0 or max_length <= 2048:
        raise ValueError("Invalid training budget")
    out = Path(output).resolve() if resume else fresh(output).resolve()
    model = str(Path(model).resolve())
    manifest = validate_data(out / "data") if resume else unpack(archive, out / "data")
    data = out / "data"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    identity = {"version": VERSION, "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                 "model": model, "seed": seed, "epochs": epochs, "max_length": max_length,
                 "manifest_hash": digest(manifest), "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest()}
    if resume:
        execution = read_json(out / "execution.json")
        if any(execution.get(k) != v for k, v in identity.items()):
            raise ValueError("Resume requires the same code commit, model, archive and training budget")
    else:
        execution = identity | {"commands": [], "phase": "preflight"}
    write_json(out / "execution.json", execution)
    try:
        preflight(data, model, max_length, out / "token_budget.json")
        command = inference_command(model, data / "direct.dev.jobs.jsonl", out / "frozen.dev.predictions.jsonl", max_length, seed)
        execution["commands"].append(execute(command, out / "logs/frozen.log"))
        for arm in ARMS:
            execution["phase"] = f"train_{arm}"
            write_json(out / "execution.json", execution)
            if not (out / arm / "completion.json").exists():
                command = train_command(data, out, arm, model, seed, epochs, max_length)
                if resume and (out / arm / "run_manifest.json").exists():
                    checkpoints = [p for p in (out / arm).glob("checkpoint-*")
                                   if p.name.split("-")[-1].isdigit() and (p / "trainer_state.json").exists()]
                    if not checkpoints:
                        raise ValueError(f"{arm}: interrupted before a resumable checkpoint; keep evidence and use a new run directory")
                    command += ["--resume", str(max(checkpoints, key=lambda p: int(p.name.split("-")[-1])))]
                execution["commands"].append(execute(command, out / f"logs/train_{arm}.log"))
            done = read_json(out / arm / "completion.json")
            artifact = Path(done["artifact"])
            if not artifact.is_dir():
                raise ValueError(f"Missing completed model artifact for {arm}")
            infer_model, adapter = (artifact, None) if done["training_mode"] == "full" else (model, artifact)
            command = inference_command(infer_model, data / f"{arm}.dev.jobs.jsonl", out / f"{arm}.dev.predictions.jsonl", max_length, seed, adapter)
            execution["commands"].append(execute(command, out / f"logs/infer_{arm}.log"))
        evaluate(data, out)
        execution["phase"] = "complete_dev"
        write_json(out / "execution.json", execution)
        write_json(out / "completion.json", {"complete": True, "stage": "dev_only", "test_evaluated": False})
    except Exception as exc:
        write_json(out / "failure.json", {"phase": execution["phase"], "error_type": type(exc).__name__, "message": str(exc)})
        write_json(out / "execution.json", execution)
        print("Failure evidence:", collect(out), flush=True)
        raise
    return {"results": collect(out), "next": "Inspect dev results before freezing the final test protocol"}


def test(run_dir):
    if platform.system() != "Linux":
        raise ValueError("Inference belongs on the GPU server")
    root = Path(run_dir).resolve()
    done = read_json(root / "completion.json")
    if not done.get("complete"):
        raise ValueError("Complete the training/dev experiment first")
    execution = read_json(root / "execution.json")
    data = root / "data"
    manifest = validate_data(data)
    if digest(manifest) != execution["manifest_hash"]:
        raise ValueError("Changed experiment inputs")
    for benchmark in manifest["tests"]:
        split = benchmark["name"]
        for arm in ("frozen",) + ARMS:
            model, adapter = execution["model"], None
            if arm != "frozen":
                training = read_json(root / arm / "completion.json")
                if training["training_mode"] == "full":
                    model = training["artifact"]
                else:
                    adapter = training["artifact"]
            command = inference_command(model, data / f"{'direct' if arm == 'frozen' else arm}.{split}.jobs.jsonl",
                root / f"{arm}.{split}.predictions.jsonl", execution["max_length"], execution["seed"], adapter)
            execute(command, root / f"logs/{arm}_{split}.log")
        evaluate(data, root, split)
    write_json(root / "test_completion.json", {"complete": True, "test_model_selection": False})
    return {"results": collect(root)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("run")
    q.add_argument("--archive", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--model", default="/mnt/yanghui/models/Qwen/Qwen3-1.7B")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--epochs", type=float, default=3)
    q.add_argument("--max-length", type=int, default=8192)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("evaluate")
    q.add_argument("--data", required=True)
    q.add_argument("--run-dir", required=True)
    q.add_argument("--split", default="dev")
    for name in ("collect", "test"):
        q = sub.add_parser(name)
        q.add_argument("--run-dir", required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    result = {"run": run, "evaluate": evaluate, "collect": collect, "test": test}[command](**args)
    if command == "evaluate":
        result = {"split": result["split"], "comparisons": result["comparisons"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
