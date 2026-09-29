"""User-run LAIC evaluation of four completed TERA arms; no training or API calls."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

from .io import digest, read_json, read_rows, write_json
from .learned import fresh
from .learned_server import collect, execute
from .server import SFT_TOKENIZATION_VERSION
from .tera_data import VERSION, prompt_features
from .tera_server import ARMS, file_hash, frozen_jobs, tokenizer_for, require_server, evaluate


def run(run_dir, output=None, max_length=32768, resume=False):
    require_server()
    source = Path(run_dir).resolve()
    data = source / "data"
    execution = read_json(source / "execution.json")
    done = read_json(source / "completion.json")
    if execution["version"] != VERSION or not done.get("complete") or done.get("phase") != "complete_dev":
        raise ValueError("Finish all four training/dev arms before LAIC evaluation")
    entries = [t for t in read_json(data / "manifest.json")["tests"] if t["path_label"] == "laic_test.jsonl"]
    if len(entries) != 1:
        raise ValueError("Expected exactly one LAIC benchmark")
    benchmark = entries[0]
    split = benchmark["name"]
    jobs = frozen_jobs(data, split)
    budget = read_json(source / "token_budget.json")
    if digest(read_rows(data / "tera.train.jsonl")) != budget["training_hash"]:
        raise ValueError("Changed prepared training data")
    config = read_json(Path(execution["model"]) / "config.json")
    if not 2048 < max_length <= config["max_position_embeddings"]:
        raise ValueError("Invalid benchmark context budget")
    identity = {"version": VERSION, "kind": "tera_laic_test", "source_run": str(source),
        "source_execution_hash": digest(execution), "training_execution": execution,
        "benchmark": benchmark, "jobs_hash": digest(jobs), "training_hash": budget["training_hash"],
        "model": execution["model"], "seed": execution["seed"], "max_length": max_length,
        "max_new_tokens": 2048, "arms": list(ARMS), "checkpoint_selection": "final",
        "test_used_for_selection": False, "training_performed": False}
    out = Path(output).resolve() if output else source / "laic_test"
    if out == source or source in out.parents and out.relative_to(source).parts[0] in ("data", *ARMS, "smoke"):
        raise ValueError("Benchmark output must not overwrite training artifacts/data")
    prior = read_json(out / "execution.json") if resume and (out / "execution.json").exists() else None
    if prior:
        if prior["identity"] != identity:
            raise ValueError("Resume benchmark identity changed")
    else:
        fresh(out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    record = {"identity": identity, "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "phase": "preflight", "commands": []}
    if prior:
        write_json(out / f"execution_before_resume_{time.time_ns()}.metrics.json", prior)
        if (out / "completion.json").exists():
            (out / "completion.json").rename(out / f"completion_before_resume_{time.time_ns()}.metrics.json")
    if (out / "failure.json").exists():
        (out / "failure.json").rename(out / f"failure_before_resume_{time.time_ns()}.metrics.json")
    write_json(out / "execution.json", record)
    try:
        tokenizer, protocol = tokenizer_for(execution["model"])
        if protocol != budget["chat_protocol"]:
            raise ValueError("Tokenizer protocol changed")
        lengths = [len(prompt_features(tokenizer, j["messages"])[0]) for j in jobs]
        overlength = [j["id"] for j, n in zip(jobs, lengths) if n + 2048 > max_length]
        write_json(out / "preflight.metrics.json", {"benchmark": benchmark, "jobs_hash": digest(jobs),
            "cases": len(jobs), "max_prompt_tokens": max(lengths), "max_length": max_length,
            "max_new_tokens": 2048, "overlength_ids": overlength, "truncation": False})
        if overlength:
            raise ValueError("LAIC context budget exceeded; no cases removed or truncated; inspect preflight.metrics.json")
        # Check all checkpoints before launching any evaluation, then copy only small provenance files.
        for arm in ARMS:
            trained = source / arm
            manifest = read_json(trained / "run_manifest.json")
            finished = read_json(trained / "completion.json")
            expected = {"version": VERSION, "arm": arm, "model": execution["model"],
                "seed": execution["seed"], "epochs": execution["epochs"], "max_steps": -1,
                "max_length": execution["max_length"], "training_hash": budget["training_hash"],
                "chat_protocol": protocol, "tokenization_version": SFT_TOKENIZATION_VERSION}
            if any(manifest.get(k) != v for k, v in expected.items()) or not finished.get("complete"):
                raise ValueError(f"Incomplete/foreign checkpoint: {arm}")
            artifact = trained / "artifact"
            checks = {p.relative_to(artifact).as_posix(): file_hash(p) for p in artifact.rglob("*") if p.is_file()}
            if (Path(finished["artifact"]).resolve() != artifact.resolve() or checks != finished["artifact_sha256"]
                    or read_json(artifact / "module.json") != manifest):
                raise ValueError(f"Changed artifact: {arm}")
            write_json(out / arm / "run_manifest.json", manifest)
            write_json(out / arm / "completion.json", finished)
        (out / "data").mkdir(exist_ok=True)
        shutil.copyfile(data / "manifest.json", out / "data/manifest.json")
        shutil.copyfile(source / "token_budget.json", out / "token_budget.json")
        for arm in ARMS:
            prediction = out / f"{arm}.{split}.predictions.jsonl"
            command = [sys.executable, "-m", "egc.tera_server", "infer", "--data", str(data),
                "--trained", str(source / arm), "--output", str(prediction), "--model", execution["model"],
                "--seed", str(execution["seed"]), "--max-length", str(max_length), "--split", split]
            if resume and prediction.exists():
                command.append("--validate-only")  # Refuse partial outputs; never silently overwrite them.
            record["phase"] = "infer_" + arm
            write_json(out / "execution.json", record)
            record["commands"].append(execute(command, out / f"logs/{arm}_{time.time_ns()}.log"))
        evaluate(data, out, split=split)
        record["phase"] = "complete_laic"
        write_json(out / "completion.json", {"complete": True, "phase": "complete_laic",
            "benchmark": benchmark, "test_evaluated": True, "test_used_for_selection": False})
    except Exception as exc:
        write_json(out / "failure.json", {"phase": record["phase"], "type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(out / "execution.json", record)
        print("Return LAIC evidence:", collect(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--resume", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
