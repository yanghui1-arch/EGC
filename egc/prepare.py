from .io import digest, index_unique
from .prompts import make_prompt, target
from .rules import validate_rules


def prepare(rows, variant, bundle=None, annotations=None, allow_draft=False):
    index_unique(rows)
    if not rows or len({r["split"] for r in rows}) != 1:
        raise ValueError("Prepare one nonempty split at a time")
    if variant != "base":
        validate_rules(bundle, allow_draft)
    lookup = index_unique(annotations or [])
    jobs, sft = [], []
    for row in rows:
        messages = make_prompt(row, variant, bundle, lookup.get(row["id"]), allow_draft)
        prompt_hash = digest(messages)
        jobs.append({"id": row["id"], "variant": variant, "split": row["split"],
                     "messages": messages, "prompt_hash": prompt_hash})
        if row["split"] != "test" and row.get("opinion"):
            sft.append({"id": row["id"], "split": row["split"], "variant": variant,
                        "dataset": row["dataset"], "source_id": row["source_id"],
                        "prompt": messages, "completion": [{"role": "assistant", "content": target(row)}]})
    manifest = {"schema_version": 1, "variant": variant, "split": rows[0]["split"],
                "dataset_hash": digest(rows), "rules_hash": digest(bundle) if bundle else None,
                "annotations_hash": digest(annotations) if annotations else None,
                "allow_draft_rules": allow_draft, "jobs_hash": digest(jobs), "n_jobs": len(jobs),
                "n_sft": len(sft), "missing_opinions_excluded_from_sft": len(rows) - len(sft) if rows[0]["split"] != "test" else 0,
                "implementation": "prompt_level_mechanism_pilot_not_ChainAwareEncoding_reproduction"}
    return jobs, sft, manifest
