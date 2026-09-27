"""Server-only 32-step native-EOS repair check; not a benchmark experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from .evaluate import parse_output
from .io import digest, read_json, read_rows, write_json, write_rows
from .learned import fresh
from .learned_server import (DEFAULT_MODEL, check_prediction_identity, collect, execute,
                             inference_command, train_command, unpack)


def select_jobs(jobs):
    selected = {}
    for job in jobs:
        if job["split"] != "dev" or job["prompt_hash"] != digest(job["messages"]):
            raise ValueError("Expected unchanged dev jobs")
        charge = json.loads(job["messages"][-1]["content"])["charge"]
        selected.setdefault(charge, job)
    if len(selected) != 12:
        raise ValueError("Expected the existing 12-charge dev cohort")
    return list(selected.values())


def run(archive, output, model=DEFAULT_MODEL):
    if platform.system() != "Linux":
        raise ValueError("Run this check only on the user's Linux GPU server")
    model = str(Path(model).resolve())
    config = read_json(Path(model) / "config.json")
    if config.get("model_type") != "qwen2" or config.get("eos_token_id") != 151643:
        raise ValueError("This repair check requires the Qwen2 Base native-EOS checkpoint")
    out = fresh(output).resolve()
    record = {"kind": "native_eos_smoke_only", "model": model, "max_steps": 32,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(), "commands": []}
    write_json(out / "execution.json", record)
    try:
        unpack(archive, out / "data")
        jobs = select_jobs(read_rows(out / "data/direct.dev.jobs.jsonl"))
        write_rows(out / "probe.jobs.jsonl", jobs)
        record["selected_ids"] = [j["id"] for j in jobs]
        command = train_command(out / "data", out, "direct", model, 42, 3, 8192) + ["--max-steps", "32"]
        record["commands"].append(execute(command, out / "logs/train_direct.log"))
        done = read_json(out / "direct/completion.json")
        if not done.get("complete") or done["training_mode"] != "lora" or done.get("global_step") != 32:
            raise ValueError("Native-EOS LoRA smoke training did not complete")
        manifest = read_json(out / "direct/run_manifest.json")
        if manifest["chat_protocol"].get("completion_end_token") != "<|endoftext|>":
            raise ValueError("Training did not use the repaired native-EOS protocol")
        prediction_file = out / "direct.dev.predictions.jsonl"
        command = inference_command(model, out / "probe.jobs.jsonl", prediction_file, 8192, 42, done["artifact"])
        record["commands"].append(execute(command, out / "logs/infer_direct.log"))
        predictions = read_rows(prediction_file)
        check_prediction_identity(jobs, predictions, read_json(str(prediction_file) + ".manifest.json"))
        valid = sum(p.get("finish_reason") == "stop" and parse_output(p["text"]) is not None for p in predictions)
        write_json(out / "completion.json", {"complete": True, "stage": "native_eos_smoke_only",
                   "n": len(jobs), "valid": valid, "termination_gate_passed": valid == len(jobs),
                   "note": "32 training steps and 12 dev generations, no MAE or method-gain conclusion"})
    except Exception as exc:
        write_json(out / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(out / "execution.json", record)
        print("Return smoke evidence:", collect(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    run(**vars(parser.parse_args()))
