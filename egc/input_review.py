"""Apply explicit, hash-bound review decisions into a NEW candidate snapshot."""
from collections import Counter

from .io import digest, index_unique


def apply_review(splits, report, queue, pairs, plan):
    if set(splits) != {"train", "dev"}:
        raise ValueError("Supply original train and dev inputs")
    if {s: digest(rows) for s, rows in splits.items()} != plan["input_hashes"]:
        raise ValueError("Review plan input hashes do not match; do not substitute inputs")
    if {"report": digest(report), "queue": digest(queue), "pairs": digest(pairs)} != plan["audit_hashes"]:
        raise ValueError("Audit evidence differs from the reviewed snapshot")
    rows = [row for split in ("train", "dev") for row in splits[split]]
    indexed = index_unique(rows)
    reviewed = index_unique(queue)
    if set(indexed) != set(reviewed):
        raise ValueError("Audit queue must cover exactly the input rows")
    for split, values in splits.items():
        if any(r["split"] != split for r in values):
            raise ValueError("Input split assignments must be preserved")
    for identifier, row in indexed.items():
        if any(reviewed[identifier].get(k) != row.get(k)
               for k in ("facts", "charge", "sentence_months", "split", "source_id")):
            raise ValueError("Audit queue row differs from input")

    prefix = plan.get("id_prefix", "")
    quarantine_reasons = {}
    for reason, identifiers in plan["quarantine_groups"].items():
        if not reason:
            raise ValueError("Every quarantine decision needs a reason")
        for short in identifiers:
            identifier = prefix + short
            if identifier not in indexed or identifier in quarantine_reasons:
                raise ValueError("Unknown or multiply assigned quarantine ID")
            quarantine_reasons[identifier] = reason
    dismissed = {prefix + k: set(v) for k, v in plan.get("dismissed_flag_kinds", {}).items()}
    for identifier, kinds in dismissed.items():
        if identifier not in indexed or identifier in quarantine_reasons or not kinds:
            raise ValueError("Invalid dismissed flag decision")
        if not kinds <= {f["kind"] for f in reviewed[identifier]["flags"]}:
            raise ValueError("Dismissal refers to a flag not present in the audit")

    pair_keys = {frozenset((p["left_id"], p["right_id"])) for p in pairs}
    links = {}
    for link in plan.get("duplicate_links", []):
        excluded, representative = prefix + link["excluded"], prefix + link["representative"]
        if (excluded == representative or excluded in links or
                quarantine_reasons.get(excluded) != "reviewed_near_duplicate" or
                representative not in indexed or frozenset((excluded, representative)) not in pair_keys):
            raise ValueError("Duplicate decision must reference an audited pair and quarantined member")
        # A pair with conflicting targets or split assignments requires a different review policy.
        if any(indexed[excluded].get(k) != indexed[representative].get(k)
               for k in ("charge", "sentence_months", "split")):
            raise ValueError("Do not resolve conflicting labels or cross-split pairs with this policy")
        links[excluded] = representative
    if {i for i, reason in quarantine_reasons.items() if reason == "reviewed_near_duplicate"} != set(links):
        raise ValueError("Every duplicate exclusion needs a representative")

    retained = {s: [r for r in values if r["id"] not in quarantine_reasons] for s, values in splits.items()}
    quarantined = [{"row": r, "reason": quarantine_reasons[r["id"]],
                    "representative_id": links.get(r["id"]), "reviewer": plan["reviewer"]}
                   for r in rows if r["id"] in quarantine_reasons]
    remaining_pairs = [p for p in pairs if p["left_id"] not in quarantine_reasons
                       and p["right_id"] not in quarantine_reasons]
    overlap_ids = {p[k] for p in remaining_pairs for k in ("left_id", "right_id")}
    pending = []
    for row in rows:
        identifier = row["id"]
        if identifier in quarantine_reasons:
            continue
        flags = [f for f in reviewed[identifier]["flags"] if f["kind"] not in dismissed.get(identifier, set())]
        if flags or identifier in overlap_ids:
            pending.append({"row": row, "remaining_flags": flags,
                            "unresolved_overlap": identifier in overlap_ids, "status": "pending_review"})
    manifest = {
        "version": "reviewed-quarantine-v1", "review_plan_version": plan["version"],
        "reviewer": plan["reviewer"], "review_plan_hash": digest(plan),
        "source_hashes": plan["input_hashes"], "audit_hashes": plan["audit_hashes"],
        "input_rows": len(rows), "candidate_rows": sum(map(len, retained.values())),
        "candidate_splits": {s: len(values) for s, values in retained.items()},
        "candidate_hashes": {s: digest(values) for s, values in retained.items()},
        "quarantined_rows": len(quarantined), "quarantine_reasons": dict(Counter(quarantine_reasons.values())),
        "quarantine_splits": dict(Counter(q["row"]["split"] for q in quarantined)),
        "pending_flagged_rows": len(pending), "remaining_audited_overlap_pairs": len(remaining_pairs),
        "candidate_charge_counts": {s: dict(Counter(r["charge"] for r in values)) for s, values in retained.items()},
        "labels_facts_ids_splits_unchanged": True, "original_inputs_untouched": True,
        "api_calls": 0, "training_ready": False,
        "limitations": ["Candidate files are not a released training set; unflagged rows are not certified clean",
                        "Quarantine includes unresolved scope and separable reasoning, not only proven bad labels",
                        "No automatic clause stripping, label repair, or resplitting",
                        "Reduced dev and selective exclusions change the evaluation population; preserve original reports",
                        "Assistant review is not an independent human gold standard"]}
    return manifest, retained, quarantined, pending
