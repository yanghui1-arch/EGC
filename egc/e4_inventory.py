"""Train-only, label-blind review inventory; lexical cues are NOT legal annotations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .io import digest, index_unique, read_json, read_rows, write_json, write_rows
from .rules import select_rules, validate_rules

VERSION = "e4-observability-inventory-v1"
TRAIN_MEMBER = "final_all_data/exercise_contest/data_train.json"
CHARGES = ("诈骗罪", "抢劫罪")
# Deliberately broad retrieval hints, including negative mentions and evidence lists.
# No match never establishes absence; a match never establishes a condition state.
CUES = {
    "arrival": r"投案|自首|抓获|传唤|归案",
    "disclosure": r"供述|交代|坦白",
    "escape": r"逃跑|逃离|逃走|脱逃",
    "retraction": r"翻供|撤回供述|推翻.{0,8}供述",
    "identity": r"身份|姓名|化名|冒名",
    "admission_repentance": r"认罪|悔罪|悔过|不持异议|没有意见",
    "restitution": r"退赃|退赔|退还|返还|偿还|赔偿|追退|追缴|发还",
    "forgiveness": r"谅解|宽恕",
    "home": r"入户|住处|住宅|居住|家中|卧室|羊圈|店内",
    "entry_purpose": r"进入|入户|预谋|临时起意|偷配|邀请|访友",
    "transport": r"出租车|客车|公交|公共汽车|班车|校车|地铁|火车|轮船|飞机",
    "date": r"(?:19|20)\d{2}年(?:\d{1,2}月)?(?:\d{1,2}日)?",
}


def cue_spans(facts):
    return [{"cue": key, "start": m.start(), "end": m.end(), "quote": m.group()}
            for key, pattern in CUES.items() for m in re.finditer(pattern, facts)]


def source_metadata(rows, source_zip):
    """Read matching train source records, retaining only identity metadata and hashes."""
    positions = defaultdict(list)
    for row in rows:
        member, number = row["source_id"].rsplit(":", 1)
        if member != TRAIN_MEMBER or not number.isdecimal() or int(number) < 1:
            raise ValueError("Expected official CAIL small train source references")
        positions[int(number)].append(row)
    result = {}
    with zipfile.ZipFile(source_zip) as archive, archive.open(TRAIN_MEMBER) as handle:
        for number, line in enumerate(handle, 1):
            if number in positions:
                raw = json.loads(line)
                for row in positions[number]:
                    if raw.get("fact") != row["facts"]:
                        raise ValueError(f"Source facts differ for {row['id']}")
                    targets = raw.get("meta", {}).get("criminals")
                    if not isinstance(targets, list) or not all(
                        isinstance(x, str) and x for x in targets
                    ):
                        raise ValueError(f"Invalid source criminals for {row['id']}")
                    result[row["id"]] = {
                        "source_record_sha256": hashlib.sha256(line).hexdigest(),
                        "facts_match_source": True,
                        "target_names_from_meta": targets,
                        "target_name_occurrences": [
                            {"name": name, "spans": [
                                {"start": m.start(), "end": m.end()}
                                for m in re.finditer(re.escape(name), row["facts"])
                            ]} for name in targets
                        ],
                        "target_binding_review": None,
                    }
            if number >= max(positions):
                break
    if len(result) != len(rows):
        raise ValueError("Some source records were not found; no inventory written")
    return result


def inventory(train, source_zip, rules, output_dir, expected_train_hash=None):
    rows = read_rows(train)
    index_unique(rows)
    if any(r.get("split") != "train" for r in rows):
        raise ValueError("Inventory only accepts train rows, never dev/test")
    if expected_train_hash is not None and digest(rows) != expected_train_hash:
        raise ValueError("Train snapshot hash mismatch")
    bundle = validate_rules(read_json(rules), allow_draft=True)
    selected = sorted((r for r in rows if r.get("charge") in CHARGES),
                      key=lambda r: (CHARGES.index(r["charge"]), r["id"]))
    if {r["charge"] for r in selected} != set(CHARGES):
        raise ValueError("Both pilot charges must have train candidates")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Use a new output directory; existing outputs are immutable")
    metadata = source_metadata(selected, source_zip)
    reviews = []
    for row in selected:
        subgraph = select_rules(bundle, row["charge"])
        # Explicit allowlist: no sentence, article labels, opinions, or raw meta copied.
        reviews.append({
            "id": row["id"], "source_id": row["source_id"], "split": "train",
            "charge": row["charge"], "facts": row["facts"],
            "facts_hash": digest(row["facts"]), "source": metadata[row["id"]],
            "cue_spans": cue_spans(row["facts"]),
            "conditions_to_review": [{
                "condition_id": c["id"], "definition": c["text"],
                "review_channel": "external_applicability" if c["id"].startswith("scope_")
                                  else "fact_and_legal_interpretation",
                "reviewed_state": None, "evidence": [], "reason": None,
            } for c in subgraph["conditions"]],
            "rule_sources": [{"rule_id": r["id"], "source": r["source"],
                              "version": r["version"]} for r in subgraph["rules"]],
            "case_applicability_review": None,
        })
    counts = Counter(cue for row in reviews for cue in {s["cue"] for s in row["cue_spans"]})
    report = {
        "version": VERSION, "input_hash": digest(rows), "rules_hash": digest(bundle),
        "review_hash": digest(reviews), "source_member": TRAIN_MEMBER,
        "input_train_rows": len(rows), "selected_rows": len(reviews),
        "charges": dict(Counter(r["charge"] for r in reviews)),
        "rows_with_cue": {key: counts[key] for key in CUES},
        "rows_without_cues": sum(not r["cue_spans"] for r in reviews),
        "rows_without_target_name_occurrence": sum(
            not any(t["spans"] for t in r["source"]["target_name_occurrences"])
            for r in reviews),
        "reviewed_conditions": 0, "verified_applicable_cases": 0,
        "api_calls": 0, "model_calls": 0,
        "limitation": "Lexical retrieval only, not condition coverage, truth, legal applicability, "
                      "or training release. None means unreviewed, not unknown. "
                      "Raw facts can contain outcome hints. Target metadata is for review only; "
                      "existing model inputs and snapshots are unchanged.",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "review_cases.jsonl", reviews)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-train-hash")
    args = parser.parse_args()
    print(json.dumps(inventory(args.train, args.source_zip, args.rules, args.output_dir,
                               args.expected_train_hash), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
