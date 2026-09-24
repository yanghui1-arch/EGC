"""Read-only provenance and conditional coverage analysis of returned result ZIPs.

No extraction, output-file writes, model imports, API requests or changes to the
original evaluation. The shared-valid cohort is a post-hoc diagnostic only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from .evaluate import compare, parse_output, summarize
from .io import digest
from .learned_server import check_prediction_identity


def checked_zip(path):
    with ZipFile(path) as z:
        checks = json.loads(z.read("checksums.json"))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(checks) | {"checksums.json"}:
            raise ValueError("Package member mismatch")
        data = {n: z.read(n) for n in checks}
    if any(hashlib.sha256(data[n]).hexdigest() != h for n, h in checks.items()):
        raise ValueError("Package checksum mismatch")
    return data


def audit(results, archive):
    data, returned = checked_zip(archive), checked_zip(results)
    rows = lambda blob: [json.loads(l) for l in blob.decode("utf-8").splitlines() if l.strip()]
    execution = json.loads(returned["execution.json"])
    manifest = json.loads(data["manifest.json"])
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != execution["archive_sha256"]:
        raise ValueError("Wrong input archive")
    if json.loads(returned["data/manifest.json"]) != manifest or digest(manifest) != execution["manifest_hash"]:
        raise ValueError("Experiment manifest mismatch")
    refs = rows(data["dev.references.jsonl"])
    saved = json.loads(returned["dev.metrics.json"])
    predictions, valid, original = {}, {}, {}
    for arm in ("frozen", "direct", "flat", "bound"):
        predictions[arm] = rows(returned[f"{arm}.dev.predictions.jsonl"])
        generation = json.loads(returned[f"{arm}.dev.predictions.jsonl.manifest.json"])
        jobs = rows(data[f"{'direct' if arm == 'frozen' else arm}.dev.jobs.jsonl"])
        check_prediction_identity(jobs, predictions[arm], generation)
        model, adapter = execution["model"], None
        if arm != "frozen":
            done = json.loads(returned[f"{arm}/completion.json"])
            train = json.loads(returned[f"{arm}/run_manifest.json"])
            if not done["complete"]:
                raise ValueError("Training incomplete")
            if done["training_mode"] == "full":
                model = done["artifact"]
            else:
                adapter = done["artifact"]
            for split in ("train", "dev"):
                if train[f"{split}_hash"] != digest(rows(data[f"{arm}.{split}.sft.jsonl"])):
                    raise ValueError("Training input identity mismatch")
        expected = {"model":model, "adapter":adapter, "seed":execution["seed"],
                    "temperature":0, "dtype":"bfloat16", "max_model_len":execution["max_length"], "max_new_tokens":2048}
        if any(generation["settings"].get(k) != v for k, v in expected.items()):
            raise ValueError("Generation configuration mismatch")
        summary = summarize(refs, predictions[arm])
        if any(summary[k] != saved["metrics"][arm][k] for k in summary):
            raise ValueError("Stored metrics do not match recomputation")
        original[arm] = {k:v for k,v in summary.items() if not isinstance(v,(list,dict))}
        valid[arm] = {r["id"] for r in summary["per_case"] if r["valid"]}
    common = set.intersection(*valid.values())
    shared_refs = [r for r in refs if r["id"] in common]
    if not shared_refs:
        raise ValueError("No shared-valid cases")
    shared_predictions = {a:[r for r in p if r["id"] in common] for a,p in predictions.items()}
    conditional, comparisons = {}, {}
    for arm,p in shared_predictions.items():
        s = summarize(shared_refs,p)
        conditional[arm] = {"n":s["n"],"mae":s["mae_months_full"],"rmse":s["rmse_months_full"],
                            "month_counts":dict(Counter(parse_output(x["text"])["sentence_months"] for x in p))}
    for first,second in (("frozen","direct"),("direct","flat"),("flat","bound")):
        comparisons[f"{second}_vs_{first}"] = compare(shared_refs,shared_predictions[first],shared_predictions[second])
    return {"provenance_and_original_metrics":"PASS", "model":execution["model"],
            "git_commit":execution["git_commit"],"retained":manifest["retained"],
            "original_metrics":original, "shared_valid_diagnostic":conditional,
            "posthoc_comparisons":comparisons,
            "excluded_from_diagnostic":{a:sorted({r['id'] for r in refs}-ids) for a,ids in valid.items()},
            "limitations":"Shared-valid metrics condition on successful generation. They do not replace the original full-cohort coverage or constitute a full-cohort gain; case bootstrap excludes training-seed uncertainty."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",default="runs/results.zip")
    p.add_argument("--archive",default="data/learned_v2_screened/experiment.zip")
    args = p.parse_args()
    print(json.dumps(audit(args.results,args.archive),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
