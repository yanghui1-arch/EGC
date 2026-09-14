from __future__ import annotations

import collections
import math
import random
import re
import unicodedata

from .io import digest, index_unique

MISSING = {"", "未知", "unknown", "null", "none"}
CIRCUMSTANCES = ("自首", "累犯", "坦白", "认罪认罚", "退赃", "谅解", "从犯", "未遂", "未成年")


def normalized_fact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def fact_hash(row):
    return digest(normalized_fact(row["facts"]))


def canonicalize(rows, dataset, split):
    """Only normalize explicit numeric month labels. Never infer labels from opinions."""
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train/dev/test")
    result = []
    for number, raw in enumerate(rows, 1):
        facts = raw.get("facts", raw.get("justice"))
        charge = raw.get("charge", raw.get("caseCause"))
        months = raw.get("sentence_months", raw.get("judge"))
        if not isinstance(facts, str) or not facts.strip():
            raise ValueError(f"Row {number}: missing facts")
        if not isinstance(charge, str) or not charge.strip():
            raise ValueError(f"Row {number}: missing given charge")
        if isinstance(months, bool) or not isinstance(months, (int, float)) or not math.isfinite(months) or months < 0 or months != int(months):
            raise ValueError(f"Row {number}: require an explicit nonnegative integer month label")
        opinion = raw.get("opinion")
        if opinion is not None and not isinstance(opinion, str):
            raise ValueError(f"Row {number}: opinion must be text or null")
        if opinion is None or opinion.strip().lower() in MISSING:
            opinion = None
        source_id = str(raw.get("source_id", raw.get("filename", raw.get("id", number))))
        row = {
            "id": f"{dataset}:{split}:{digest([source_id, facts])[:20]}",
            "dataset": dataset, "split": split, "source_id": source_id,
            "facts": facts, "charge": charge.strip(), "opinion": opinion,
            "sentence_months": int(months), "protocol": "given_charge",
        }
        row["input_hash"] = digest(visible_case(row))
        result.append(row)
    index_unique(result)
    return result


def visible_case(row):
    """Allowlist is the only source for model prompts/annotation/cache keys."""
    return {"facts": row["facts"], "charge": row["charge"]}


def audit(rows):
    index_unique(rows)
    duplicates = collections.defaultdict(list)
    review = []
    for row in rows:
        duplicates[fact_hash(row)].append(row["id"])
        # Indicators only: prior convictions may legitimately appear in facts.
        possible_outcome = bool(re.search(r"判处.{0,16}(有期徒刑|拘役|无期徒刑|死刑)|判决如下", row["facts"]))
        absent_terms = [term for term in CIRCUMSTANCES if term in (row["opinion"] or "") and term not in row["facts"]]
        if possible_outcome or absent_terms:
            review.append({"id": row["id"], "possible_outcome_in_input": possible_outcome,
                           "opinion_only_keywords": absent_terms,
                           "interpretation": "heuristic_review_required_not_a_gold_label"})
    return {
        "n": len(rows), "dataset_hash": digest(rows),
        "splits": dict(collections.Counter(r["split"] for r in rows)),
        "charges": dict(collections.Counter(r["charge"] for r in rows)),
        "missing_opinion": sum(r["opinion"] is None for r in rows),
        "zero_month_labels_to_review": sum(r["sentence_months"] == 0 for r in rows),
        "exact_duplicate_groups": [ids for ids in duplicates.values() if len(ids) > 1],
        "review_flags": review,
        "limitations": "No automatic legal sufficiency judgment; near-duplicates need manual/source-level review.",
    }


def split_train(rows, dev_ratio=0.1, seed=42):
    index_unique(rows)
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Refusing to split a dev/test set into training data")
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")
    # Connected groups by identical facts OR source case ID cannot cross splits.
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in [("facts", fact_hash(row)), ("source", row["dataset"], row["source_id"])]:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    strata = collections.defaultdict(list)
    for group in groups.values():
        key = tuple(sorted({r["charge"] for r in group}))
        strata[key].append(group)
    rng = random.Random(seed)
    train, dev = [], []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda g: min(r["id"] for r in g))
        rng.shuffle(bucket)
        n = min(len(bucket) - 1, max(1, round(len(bucket) * dev_ratio))) if len(bucket) > 1 else 0
        for i, group in enumerate(bucket):
            for row in sorted(group, key=lambda r: r["id"]):
                copy = dict(row, split="dev" if i < n else "train")
                (dev if i < n else train).append(copy)
    if not train or not dev:
        raise ValueError("Not enough independent source groups for train/dev split")
    return train, dev


def overlap_report(named_rows):
    facts, sources = collections.defaultdict(list), collections.defaultdict(list)
    for name, rows in named_rows.items():
        for row in rows:
            facts[fact_hash(row)].append((name, row["id"]))
            sources[(row["dataset"], row["source_id"])].append((name, row["id"]))
    def cross(groups):
        return [items for items in groups.values() if len({n for n, _ in items}) > 1]
    return {"exact_fact_overlap": cross(facts), "source_overlap": cross(sources)}


def pilot_review(rows, n=200, seed=42):
    if any(r["split"] == "test" for r in rows):
        raise ValueError("Pilot selection must use train/dev, not held-out test")
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        buckets[row["charge"]].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < min(n, len(rows)):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop())
    return [dict(r, review={"input_sufficiency": None, "wrong_base_range": None,
                            "missed_supported_modifier": None, "condition_error": None,
                            "reviewer": None, "notes": None}) for r in selected]
