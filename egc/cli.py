from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data
from .io import read_json, read_rows, write_json, write_rows


def run(args):
    cmd = args.command
    if cmd == "facts-prepare":
        from .fact_experiment import prepare
        report = prepare(read_rows(args.pool), [r for p in args.exclude for r in read_rows(p)],
                         read_json(args.regression), read_json(args.profile), args.output_dir,
                         args.per_charge, args.seed)
        return {k: report[k] for k in ("regression_cases", "new_cases", "new_counts", "requested_per_charge")}
    if cmd == "facts-run":
        from .fact_experiment import run as run_facts
        return run_facts(read_rows(args.input), read_json(args.profile), args.output_dir,
                         args.modes, args.model, args.limit, args.max_tokens, args.ask_key, args.dry_run,
                         read_json(args.protocol) if args.protocol else None)
    if cmd == "facts-report":
        from .fact_experiment import report_run
        report = report_run(read_rows(args.input), args.run_dir, args.output,
                            read_json(args.expectations) if args.expectations else None)
        return {"output": args.output, "by_mode": report["by_mode"],
                "regression_agreement_not_accuracy": {
                    k: {f: v[f] for f in ("matched", "total")}
                    for k, v in report.get("regression_agreement_not_accuracy", {}).items()},
                "semantic_accuracy": None, "training_ready": False}
    if cmd == "audit-annotations":
        from .annotation_audit import audit
        report, queue = audit(read_rows(args.source), read_rows(args.annotations), read_json(args.profile),
                              args.sample_size, args.seed)
        out = Path(args.output_dir)
        write_json(out / "report.json", report)
        write_rows(out / "review_queue.jsonl", queue)
        return {k: report[k] for k in ("source_cases", "structurally_accepted_cases", "flagged_cases",
                "review_queue_cases", "semantic_accuracy", "training_ready")}
    if cmd == "build-cail":
        from .cail import build
        return build(args.archive, args.member, args.benchmarks, args.output_dir, args.per_charge, args.seed)
    if cmd == "distill":
        from .distill import run as distill
        return distill(read_rows(args.input), read_json(args.profile), args.output_dir,
                       args.model, args.limit, args.max_tokens, args.ask_key, args.dry_run, args.transport)
    if cmd == "normalize":
        rows = data.canonicalize(read_rows(args.input), args.dataset, args.split)
        write_rows(args.output, rows)
        write_json(str(args.output) + ".audit.json", data.audit(rows))
        return {"rows": len(rows), "output": args.output}
    if cmd == "audit":
        result = data.audit(read_rows(args.input))
        write_json(args.output, result)
        return {"n": result["n"], "flags": len(result["review_flags"]), "output": args.output}
    if cmd == "split":
        train, dev = data.split_train(read_rows(args.input), args.dev_ratio, args.seed)
        out = Path(args.output_dir)
        write_rows(out / "train.jsonl", train)
        write_rows(out / "dev.jsonl", dev)
        write_json(out / "split_manifest.json", {"seed": args.seed, "dev_ratio": args.dev_ratio,
                    "train": data.audit(train), "dev": data.audit(dev),
                    "overlap": data.overlap_report({"train": train, "dev": dev})})
        return {"train": len(train), "dev": len(dev)}
    if cmd == "overlap":
        result = data.overlap_report({p: read_rows(p) for p in args.inputs})
        write_json(args.output, result)
        if any(result.values()):
            raise ValueError("Cross-file overlap found; inspect report before training")
        return {"overlap": 0, "report": args.output}
    if cmd == "pilot":
        rows = data.pilot_review(read_rows(args.input), args.n, args.seed)
        write_rows(args.output, rows)
        return {"selected": len(rows), "requested": args.n}
    if cmd == "import-chains":
        from .rules import import_chains
        result = import_chains(Path(args.input).read_text(encoding="utf-8-sig"), args.charge, args.source, args.version)
        write_json(args.output, result)
        return {"draft_rules": len(result["rules"]), "review_required": True}
    if cmd == "annotate":
        from .annotate import annotate
        return annotate(read_rows(args.input), read_json(args.rules), args.output, args.cache_dir,
                        args.model, args.limit, args.max_tokens, args.dry_run, args.allow_draft)
    if cmd == "prepare":
        from .prepare import prepare
        jobs, sft, manifest = prepare(read_rows(args.input), args.variant,
                                     read_json(args.rules) if args.rules else None,
                                     read_rows(args.annotations) if args.annotations else None, args.allow_draft)
        out = Path(args.output_dir)
        if out.exists() and any(out.iterdir()):
            raise ValueError("Prepared output directory must be new (avoid mixing experiment variants)")
        write_rows(out / "jobs.jsonl", jobs)
        if sft:
            write_rows(out / "sft.jsonl", sft)
        write_json(out / "manifest.json", manifest)
        return manifest
    if cmd == "evaluate":
        from .evaluate import summarize
        result = summarize(read_rows(args.reference), read_rows(args.predictions))
        write_json(args.output, result)
        return {k: v for k, v in result.items() if k != "per_case"}
    if cmd == "compare":
        from .evaluate import compare
        result = compare(read_rows(args.reference), read_rows(args.baseline), read_rows(args.candidate), args.seed, args.samples)
        write_json(args.output, result)
        return result
    if cmd == "fetch-upstream":
        from .fetch import fetch_upstream
        result = fetch_upstream(args.manifest, args.output_dir)
        return {"revision": result["revision"], "files": len(result["files"])}
    if cmd == "doctor":
        from .server import environment
        result = environment(args.model_root, args.gpu)
        write_json(args.output, result)
        return result
    if cmd in {"train", "infer"}:
        from .server import train, infer
        (train if cmd == "train" else infer)(args)
        return {"complete": True, "command": cmd}
    raise ValueError("Unknown command")


def parser():
    p = argparse.ArgumentParser(description="EGC evidence/chain pilot. CPU preprocessing; GPU commands run on server.")
    subs = p.add_subparsers(dest="command", required=True)
    def command(name, description):
        return subs.add_parser(name, help=description)
    def io(q):
        q.add_argument("--input", required=True)
        q.add_argument("--output", required=True)
    q = command("facts-prepare", "USER RUN: freeze E2 regression and unexposed train cohorts; no API")
    q.add_argument("--pool", required=True)
    q.add_argument("--exclude", nargs="+", required=True)
    q.add_argument("--regression", default="configs/fact_regression.json")
    q.add_argument("--profile", default="configs/evidence_profile.json")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--per-charge", type=int, default=30)
    q.add_argument("--seed", type=int, default=42)
    q = command("facts-run", "USER RUN: v1/v2 fact extraction, request-bounded cached curl API")
    q.add_argument("--input", required=True)
    q.add_argument("--profile", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--modes", nargs="+", choices=["joint", "flat", "bound", "flat_v2", "bound_v2", "flat_e3", "bound_e3"], default=["joint", "flat", "bound"])
    q.add_argument("--protocol", help="Required frozen cohort/config protocol for E3")
    q.add_argument("--model", default="deepseek-flash")
    q.add_argument("--limit", type=int, default=18, help="Maximum new HTTP requests, not cases; curl never retries")
    q.add_argument("--max-tokens", type=int, default=4096)
    q.add_argument("--ask-key", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q = command("facts-report", "USER RUN: compare raw-validated results; expectations never sent to API")
    q.add_argument("--input", required=True)
    q.add_argument("--run-dir", required=True)
    q.add_argument("--expectations")
    q.add_argument("--output", required=True)
    q = command("audit-annotations", "Check original labels/evidence; sample semantic review queue")
    q.add_argument("--source", required=True)
    q.add_argument("--annotations", required=True)
    q.add_argument("--profile", default="configs/evidence_profile.json")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--sample-size", type=int, default=30)
    q.add_argument("--seed", type=int, default=42)
    q = command("build-cail", "Build original-label training pool with held-out overlap screening")
    q.add_argument("--archive", required=True)
    q.add_argument("--member", required=True)
    q.add_argument("--benchmarks", nargs="+", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--per-charge", type=int, default=1000)
    q.add_argument("--seed", type=int, default=42)
    q = command("distill", "Source-only DeepSeek evidence/analysis drafts; preserve original labels")
    q.add_argument("--input", required=True)
    q.add_argument("--profile", default="configs/evidence_profile.json")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--model", default="deepseek-flash")
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--max-tokens", type=int, default=4096)
    q.add_argument("--ask-key", action="store_true")
    q.add_argument("--transport", choices=["urllib", "curl"], default="urllib")
    q.add_argument("--dry-run", action="store_true")
    q = command("normalize", "Normalize upstream JSON(L), preserve facts, mark train/dev/test")
    io(q)
    q.add_argument("--dataset", required=True)
    q.add_argument("--split", choices=["train", "dev", "test"], required=True)
    q = command("audit", "Create heuristic input/reference review flags")
    io(q)
    q = command("split", "Split TRAIN only, group by source case and identical facts")
    q.add_argument("--input", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--dev-ratio", type=float, default=0.1)
    q.add_argument("--seed", type=int, default=42)
    q = command("overlap", "Fail on cross-file identical facts or same-source cases")
    q.add_argument("--inputs", nargs="+", required=True)
    q.add_argument("--output", required=True)
    q = command("pilot", "Select balanced review cases from train/dev, never test")
    io(q)
    q.add_argument("--n", type=int, default=200)
    q.add_argument("--seed", type=int, default=42)
    q = command("import-chains", "Import original chain text as draft rules requiring review")
    io(q)
    q.add_argument("--charge", required=True)
    q.add_argument("--source", required=True)
    q.add_argument("--version", required=True)
    q = command("annotate", "DeepSeek source-only condition annotation with cache and request bound")
    io(q)
    q.add_argument("--rules", required=True)
    q.add_argument("--model", required=True, help="Explicit available DeepSeek model ID; recorded for reproducibility")
    q.add_argument("--cache-dir", default="data/cache/deepseek")
    q.add_argument("--limit", type=int, default=20, help="Maximum new cases per invocation; up to 3 transient HTTP attempts per case")
    q.add_argument("--max-tokens", type=int, default=4096)
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--allow-draft", action="store_true")
    q = command("prepare", "Create label-free inference jobs and separate train/dev SFT pairs")
    q.add_argument("--input", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--variant", choices=["base", "rules", "concat", "egc"], required=True)
    q.add_argument("--rules")
    q.add_argument("--annotations")
    q.add_argument("--allow-draft", action="store_true")
    q = command("evaluate", "MAE/RMSE, strict parse coverage, diagnostic character ROUGE")
    q.add_argument("--reference", required=True)
    q.add_argument("--predictions", required=True)
    q.add_argument("--output", required=True)
    q = command("compare", "Paired bootstrap MAE difference; full coverage required")
    q.add_argument("--reference", required=True)
    q.add_argument("--baseline", required=True)
    q.add_argument("--candidate", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--samples", type=int, default=2000)
    q = command("fetch-upstream", "Fetch pinned public resources and verify Git blob checksums")
    q.add_argument("--manifest", default="configs/upstream.json")
    q.add_argument("--output-dir", default="data/raw/upstream")
    q = command("doctor", "Report server package versions, model directories, optional CUDA information")
    q.add_argument("--model-root", default="/mnt/yanghui/models/Qwen")
    q.add_argument("--gpu", action="store_true")
    q.add_argument("--output", default="runs/doctor.json")
    q = command("train", "SERVER ONLY: <7B full SFT; >=7B LoRA; dev checkpoint selection")
    q.add_argument("--model", required=True)
    q.add_argument("--training-mode", choices=["auto", "full", "lora"], default="auto",
                   help="Auto uses actual total parameter count; explicit mode must obey the <7B full / >=7B LoRA policy")
    q.add_argument("--train", required=True)
    q.add_argument("--dev", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--max-length", type=int, default=4096)
    q.add_argument("--batch-size", type=int, default=1)
    q.add_argument("--grad-accum", type=int, default=16)
    q.add_argument("--rank", type=int, default=16)
    q.add_argument("--epochs", type=float, default=3)
    q.add_argument("--learning-rate", type=float, default=1e-5)
    q.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    q.add_argument("--eval-steps", type=int, default=100)
    q.add_argument("--max-steps", type=int, default=-1)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--resume")
    q = command("infer", "SERVER ONLY: resumable vLLM inference, optional trained LoRA adapter")
    q.add_argument("--model", required=True)
    q.add_argument("--adapter")
    q.add_argument("--jobs", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--max-model-len", type=int, default=8192)
    q.add_argument("--max-new-tokens", type=int, default=1024)
    q.add_argument("--tensor-parallel", type=int, default=1)
    q.add_argument("--gpu-memory", type=float, default=0.85)
    q.add_argument("--temperature", type=float, default=0.0)
    q.add_argument("--seed", type=int, default=42)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "facts-run" and any(result.get("statuses", {}).get(s, 0) for s in ("api_error", "rejected")):
            p.exit(3, "EGC: Fact run contains failed requests; inspect manifest/raw diagnostics before continuing.\n")
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        p.exit(2, f"EGC: {exc}\n")


if __name__ == "__main__":
    main()
