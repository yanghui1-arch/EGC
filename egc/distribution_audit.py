"""Read-only aggregate audit of the current training package and three benchmarks.

Run from the repository root: python -m egc.distribution_audit
No API/model calls, resampling, data writes, or prediction-score changes.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from .data import canonicalize, fact_hash
from .io import digest, index_unique, read_json, read_rows
from .learned import messages, visible
from .learned_audit import checked_zip


BINS = ("0", "1-6", "7-12", "13-36", "37-60", "61-120", ">120")


def month_bin(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected original nonnegative integer months")
    return next((name for name, ceiling in zip(BINS, (0, 6, 12, 36, 60, 120))
                 if value <= ceiling), ">120")


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def numeric(values):
    return {"mean": mean(values), "min": min(values), "p10": quantile(values, .1),
            "p25": quantile(values, .25), "median": median(values),
            "p75": quantile(values, .75), "p90": quantile(values, .9), "max": max(values)}


def profile(rows):
    index_unique(rows)
    months = [r["sentence_months"] for r in rows]
    bins = Counter(month_bin(m) for m in months)
    lengths = [len(r["facts"]) for r in rows]
    charges = sorted({r["charge"] for r in rows})
    return {"n": len(rows), "rows_hash": digest(rows), "charge_counts": dict(Counter(r["charge"] for r in rows)),
            "months": numeric(months), "month_counts": dict(sorted(Counter(months).items())),
            "month_bins": {k: bins[k] for k in BINS},
            "months_12_36_120_n": sum(m in (12, 36, 120) for m in months),
            "facts_chars": numeric(lengths), "facts_over_2400_n": sum(n > 2400 for n in lengths),
            "explicit_target_n": sum(bool(r.get("target_person")) for r in rows),
            "missing_opinion_n": sum(not r.get("opinion") for r in rows),
            "duplicate_normalized_fact_rows": len(rows) - len({fact_hash(r) for r in rows}),
            "by_charge": {c: {"n": sum(r["charge"] == c for r in rows),
                              "months": numeric([r["sentence_months"] for r in rows if r["charge"] == c])}
                          for c in charges}}


def total_variation(first, second):
    return sum(abs(first[k] / sum(first.values()) - second[k] / sum(second.values()))
               for k in first.keys() | second.keys()) / 2


def audit():
    archive = Path("data/learned_v2_screened/experiment.zip")
    files = checked_zip(archive)
    rows = lambda blob: [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    manifest = json.loads(files["manifest.json"])
    datasets = {s: rows(files[f"{s}.references.jsonl"]) for s in ("train", "dev")}
    for test in manifest["tests"]:
        data = rows(files[f"{test['name']}.references.jsonl"])
        assert len(data) == test["n"] and digest(data) == test["hash"]
        name = test["path_label"].removesuffix("_test.jsonl")
        upstream = read_rows(Path("data/raw/upstream/data") / name.upper() / "test_data.json")
        assert canonicalize(upstream, data[0]["dataset"], "test") == data
        assert all(set(r) == {"caseCause", "filename", "judge", "justice", "opinion", "province"} for r in upstream)
        datasets[name] = data
        jobs = rows(files[f"direct.{test['name']}.jobs.jsonl"])
        assert len(jobs) == len(data)
        for row, job in zip(data, jobs):
            assert job == {"id": row["id"], "split": "test", "variant": "direct",
                           "messages": messages(row, "direct"), "prompt_hash": digest(messages(row, "direct"))}
    assert set(datasets) == {"train", "dev", "laic", "pccd", "cail"}
    for split in ("train", "dev"):
        assert len(datasets[split]) == manifest["retained"][split]
        for arm in ("direct", "flat", "bound"):
            sft = rows(files[f"{arm}.{split}.sft.jsonl"])
            assert [r["id"] for r in sft] == [r["id"] for r in datasets[split]]
            for reference, item in zip(datasets[split], sft):
                answer = json.loads(item["completion"][0]["content"])
                assert type(answer["sentence_months"]) is int
                assert answer["sentence_months"] == reference["sentence_months"]
                assert item["prompt"] == messages(reference, arm)
                assert json.loads(item["prompt"][-1]["content"]) == visible(reference)
    train = datasets["train"]
    profiles = {name: profile(data) for name, data in datasets.items()}
    comparisons = {}
    for name, data in datasets.items():
        if name == "train":
            continue
        charge = Counter(r["charge"] for r in data)
        assert set(charge) <= set(profiles["train"]["charge_counts"])
        reweighted_mean = sum(count / len(data) * profiles["train"]["by_charge"][c]["months"]["mean"]
                              for c, count in charge.items())
        comparisons[name] = {
            "charge_tvd": total_variation(Counter(profiles["train"]["charge_counts"]), charge),
            "month_bin_tvd": total_variation(Counter(profiles["train"]["month_bins"]), Counter(profiles[name]["month_bins"])),
            "training_mean_at_evaluation_charge_mix": reweighted_mean,
            "evaluation_mean_minus_charge_reweighted_training_mean": profiles[name]["months"]["mean"] - reweighted_mean,
            "facts_longer_than_training_max_n": sum(len(r["facts"]) > profiles["train"]["facts_chars"]["max"] for r in data),
            "exact_normalized_fact_overlap_with_train": len({fact_hash(r) for r in train} & {fact_hash(r) for r in data})}
    pool = read_rows("data/learned_v1/pool/train.jsonl")
    pool_manifest = read_json("data/learned_v1/pool/manifest.json")
    assert digest(pool) == pool_manifest["train_hash"]
    prior_archive = Path("data/learned_v2/ready/experiment.zip")
    prior = checked_zip(prior_archive)
    assert hashlib.sha256(prior_archive.read_bytes()).hexdigest() == manifest["screening"]["source_archive_sha256"]
    kept = rows(prior["train.references.jsonl"])
    pool_by_id, kept_by_id = index_unique(pool), index_unique(kept)
    assert all(r == pool_by_id[r["id"]] for r in kept)
    assert all(r == kept_by_id[r["id"]] for r in train)
    for test in manifest["tests"]:
        member = f"{test['name']}.references.jsonl"
        assert files[member] == prior[member]
    aligned = json.loads(checked_zip("runs/results_1790624869321104843.zip")["token_budget.json"])
    assert aligned["cases"] == len(train)
    role_counts = {k: aligned["alignment"][k] for k in ("target", "other", "uncertain")}
    return {"source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "checks": "PASS: ZIP checksums, IDs, upstream benchmark normalization/hashes/schema, canonical inputs, SFT months, stage row identity",
            "profiles": profiles, "versus_train": comparisons,
            "training_selection_stages": {"pool": profile(pool), "teacher_keep": profile(kept), "screened": profiles["train"]},
            "auxiliary_role_counts": role_counts,
            "auxiliary_role_shares": {k: v / sum(role_counts.values()) for k, v in role_counts.items()},
            "auxiliary_unlabelled_or_conflict_n": aligned["alignment"]["unlabelled"] + aligned["alignment"]["conflict"],
            "limitations": ["Characters, not tokens; missing target field does not mean no person in facts",
                            "Test labels inspected for diagnosis, not an untouched final holdout thereafter",
                            "Marginal distributions do not establish causality or benefit from resampling",
                            "Exact normalized overlaps do not rule out semantic/source-case overlap",
                            "CAIL zero-month label meaning and weak-role semantic accuracy remain unverified"]}


if __name__ == "__main__":
    # Small deterministic checks for the numerical helpers, before real-data audit assertions.
    assert [month_bin(m) for m in (0, 1, 6, 7, 12, 13, 36, 37, 60, 61, 120, 121)] == [
        "0", "1-6", "1-6", "7-12", "7-12", "13-36", "13-36", "37-60", "37-60", "61-120", "61-120", ">120"]
    assert quantile([0, 10], .25) == 2.5
    assert total_variation(Counter(a=1), Counter(a=1)) == 0
    assert total_variation(Counter(a=1), Counter(b=1)) == 1
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
