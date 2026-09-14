"""Offline software integration check; never a model/benchmark experiment."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from egc.data import canonicalize, split_train
from egc.evaluate import summarize
from egc.io import read_json, read_rows, write_json, write_rows
from egc.prepare import prepare
from egc.rules import resolve_quotes, select_rules
from egc.server import check_training_rows


def smoke(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new smoke output directory")
    rules = read_json(ROOT / "examples/synthetic_rules.json")
    train, dev = split_train(canonicalize(read_rows(ROOT / "examples/synthetic_train.jsonl"), "synthetic", "train"), .2, 42)
    test = canonicalize(read_rows(ROOT / "examples/synthetic_test.jsonl"), "synthetic", "test")
    prepared = {}
    for split, rows in (("train", train), ("dev", dev), ("test", test)):
        annotations = []
        for row in rows:
            returned = "已返还合成物品。" in row["facts"]
            annotations.append(resolve_quotes({"conditions": [
                {"condition_id": "action", "status": "supported", "quotes": ["实施了合成行为A"]},
                {"condition_id": "returned", "status": "supported" if returned else "unknown",
                 "quotes": ["已返还合成物品。"] if returned else []}]}, row, select_rules(rules, row["charge"])))
        for variant in ("base", "rules", "concat", "egc"):
            jobs, sft, manifest = prepare(rows, variant, rules, annotations)
            out = directory / variant / split
            write_rows(out / "jobs.jsonl", jobs)
            write_json(out / "manifest.json", manifest)
            if sft:
                write_rows(out / "sft.jsonl", sft)
            prepared[variant, split] = sft
    for variant in ("base", "rules", "concat", "egc"):
        check_training_rows(prepared[variant, "train"], prepared[variant, "dev"])
        assert not prepared[variant, "test"]
    # Deliberately fixed dummy outputs; do not report them as model predictions.
    predictions = [{"id": row["id"], "text": json.dumps({"reasoning": "虚构软件占位输出，无模型参与。", "sentence_months": 1})} for row in test]
    report = summarize(test, predictions)
    assert report["coverage"] == 1
    write_json(directory / "software_check.json", {"software_only": True, "model_calls": 0,
               "api_calls": 0, "variants": 4, "train": len(train), "dev": len(dev), "test": len(test)})
    print("PASS: offline software integration, 4 variants; no model/API calls; no benchmark scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Keep synthetic jobs for a SERVER smoke run; must be new")
    args = parser.parse_args()
    if args.output_dir:
        smoke(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="egc-smoke-") as temporary:
            smoke(temporary)
