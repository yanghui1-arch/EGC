"""Reproducible integrity checks and review sampling; never an accuracy estimator."""
from collections import Counter
import random
import re

from .data import visible_case
from .cail import input_risks
from .io import digest, index_unique
from .rules import validate_annotation, resolve_quotes


def risk_flags(row):
    conditions = row["distillation"]["condition_annotation"]["conditions"]
    flags = input_risks(row["facts"])
    cues = {"voluntary_appearance": r"抓获|自动投案|主动.{0,12}投案",
            "truthful_confession": r"如实供述",
            "returned_or_compensated": r"退赃|退出赃款|退还|赔偿|补缴",
            "victim_forgiveness": r"谅解",
            "prior_conviction": r"曾因.{0,30}判",
            "minor_at_offence": r"未满十八|未满18",
            "secondary_role": r"从犯|次要作用|辅助作用",
            "unsuccessful_completion": r"未得逞|未遂"}
    for item in conditions:
        cid, status = item["condition_id"], item["status"]
        quote = " ".join(s["quote"] for s in item.get("evidence", []))
        if status == "unknown" and re.search(cues.get(cid, r"(?!x)x"), row["facts"]):
            flags.append("explicit_cue_unknown_review:" + cid)
        if cid == "voluntary_appearance" and status == "refuted" and "传唤" in quote:
            flags.append("summons_refuted_review")
        if cid == "returned_or_compensated" and status == "supported":
            if re.search(r"公司|单位|借款人|追回|追缴|扣押|收缴", quote):
                flags.append("compensation_actor_review")
        if cid == "unsuccessful_completion" and status == "supported":
            if re.search(r"多次|分别|第[一二三四五六七八九十]次|[123456789]、", row["facts"]):
                flags.append("multiple_events_review")
        if cid == "minor_at_offence" and status == "supported":
            flags.append("age_actor_and_date_review")
        if cid == "secondary_role" and status == "refuted":
            flags.append("role_negation_review")
        if status == "unknown" and item.get("evidence"):
            flags.append("unknown_with_evidence_review")
    return sorted(set(flags))


def audit(source, annotated, profile, sample_size=30, seed=42):
    if sample_size <= 0:
        raise ValueError("Review sample size must be positive")
    originals = index_unique(source)
    index_unique(annotated)
    if not source or not annotated:
        raise ValueError("Expected nonempty sources and annotations")
    states, by_condition, charges = Counter(), {}, Counter()
    flags_by_id = {}
    for row in annotated:
        original = originals.get(row["id"])
        if original is None or visible_case(row) != visible_case(original):
            raise ValueError("Annotated input is missing or changed from original source")
        if any(row.get(k) != original.get(k) for k in ("sentence_months", "split", "dataset")):
            raise ValueError("Original label/split/dataset changed")
        if row["split"] not in {"train", "dev"}:
            raise ValueError("Training annotation audit refuses test data")
        provenance = row.get("opinion_provenance", {})
        if (provenance.get("kind") != "synthetic_source_only" or
                provenance.get("reference_labels_sent_to_teacher") is not False):
            raise ValueError("Missing source-only synthetic provenance")
        result = row["distillation"]
        if row["opinion"] != result["synthetic_opinion"]:
            raise ValueError("Synthetic target differs from recorded annotation")
        if row["opinion"] != "\n".join(c["text"] for c in result["claims"]):
            raise ValueError("Synthetic target differs from cited claims")
        validate_annotation(result["condition_annotation"], row, profile)
        for claim in result["claims"]:
            resolved = resolve_quotes({"conditions": [{"condition_id": "claim", "status": "supported",
                "quotes": [s["quote"] for s in claim["evidence"]]}]}, row,
                {"conditions": [{"id": "claim", "text": claim["text"]}]})
            if resolved["conditions"][0]["evidence"] != claim["evidence"]:
                raise ValueError("Claim evidence offsets changed")
        charges[row["charge"]] += 1
        for condition in result["condition_annotation"]["conditions"]:
            cid, status = condition["condition_id"], condition["status"]
            states[status] += 1
            by_condition.setdefault(cid, Counter())[status] += 1
        flags = risk_flags(row)
        if flags:
            flags_by_id[row["id"]] = flags
    # Sorted identifiers make the draw independent of annotation completion order.
    ids = sorted(row["id"] for row in annotated)
    sampled = random.Random(seed).sample(ids, min(sample_size, len(ids)))
    selected = set(sampled) | flags_by_id.keys()
    indexed = index_unique(annotated)
    queue = [{"id": identifier, "selection": {
        "random_sample": identifier in sampled, "heuristic_flags": flags_by_id.get(identifier, [])},
        "facts": indexed[identifier]["facts"], "charge": indexed[identifier]["charge"],
        "synthetic_opinion": indexed[identifier]["opinion"],
        "distillation": indexed[identifier]["distillation"],
        "review_status": "pending_model_or_human_review"}
        for identifier in sorted(selected)]
    report = {"source_cases": len(source), "structurally_accepted_cases": len(annotated),
        "missing_source_ids": sorted(originals.keys() - set(ids)),
        "source_hash": digest(source), "annotations_hash": digest(annotated), "profile_hash": digest(profile),
        "original_input_label_and_split_integrity": "passed", "exact_evidence_integrity": "passed",
        "charges": dict(sorted(charges.items())), "condition_states": dict(states),
        "by_condition": {k: dict(v) for k, v in sorted(by_condition.items())},
        "heuristic_flag_counts": dict(Counter(f for flags in flags_by_id.values() for f in flags)),
        "flagged_cases": len(flags_by_id), "random_sample_ids": sampled, "seed": seed,
        "review_queue_cases": len(queue), "semantic_accuracy": None,
        "training_ready": False, "limitation": "Flags select review targets, not confirmed errors. "
        "Exact quotes and unchanged labels do not establish semantic or legal correctness."}
    return report, queue
