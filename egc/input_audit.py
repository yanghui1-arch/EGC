"""Review-only input QA. Never edits labels, facts, splits, or legacy experiment results."""
from collections import Counter, defaultdict
import re

from .cail import grams, input_risks
from .data import normalized_fact
from .io import digest

VERSION = "input-review-v3"
PATTERNS = {
    "possible_sentence_hint": re.compile(
        r"(?:建议[^。；\n]{0,60}?判处|判处)[^。；\n]{0,60}?"
        r"(?:[零〇一二三四五六七八九十百两0-9]{1,8}(?:个)?(?:月|年)|有期徒刑|拘役|无期徒刑|死刑)"),
    "possible_reasoning_text": re.compile(
        r"(?:从轻|减轻|从重|免予)[^。；\n]{0,12}(?:处罚|判处)|"
        r"量刑[^。；\n]{0,20}(?:考虑|采纳)|(?:不予|予以)采纳|"
        r"(?:系|是|属)[^。；\n]{0,6}(?:主犯|从犯|累犯|自首|(?:犯罪)?未遂)"),
    "possible_multiple_charge_scope": re.compile(r"数罪并罚|犯[二两三四0-9]+个罪|罪[、，][^。；\n]{1,12}罪"),
}


def review_flags(text):
    flags = [{"kind": kind, "start": None, "end": None, "quote": None} for kind in input_risks(text)]
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            flags.append({"kind": kind, "start": match.start(), "end": match.end(), "quote": match.group()})
    return flags


def audit(named_rows, focus=None, threshold=.85):
    if not 0 < threshold <= 1:
        raise ValueError("Near-overlap threshold must be in (0, 1]")
    entries = [(name, row) for name, rows in named_rows.items() for row in rows]
    if not entries:
        raise ValueError("No inputs to audit")
    # Inverted index checks ALL earlier rows, including the same split/cohort.
    postings, exact = defaultdict(set), defaultdict(list)
    groups, texts, pairs = [], [], []
    reviews, counts = [], Counter()
    for i, (name, row) in enumerate(entries):
        text = normalized_fact(row["facts"])
        group = grams(row["facts"])
        shared = Counter(j for gram in group for j in postings.get(gram, ()))
        candidates = set(shared) | set(exact.get(text, []))
        for j in sorted(candidates):
            denominator = min(len(group), len(groups[j]))
            is_exact = text == texts[j]
            score = 1.0 if is_exact else shared[j] / denominator if denominator >= 20 else 0.0
            if score < threshold:
                continue
            other_name, other = entries[j]
            pairs.append({"left_index": j, "right_index": i, "left_id": other["id"], "right_id": row["id"],
                          "left_input": other_name, "right_input": name,
                          "method": "normalized_exact" if is_exact else "character_5gram_containment",
                          "score": round(score, 4), "cross_split": other["split"] != row["split"],
                          "labels_differ": (other["charge"], other.get("sentence_months")) !=
                                           (row["charge"], row.get("sentence_months")),
                          "review_status": "candidate_not_confirmed_same_case"})
        for gram in group:
            postings[gram].add(i)
        exact[text].append(i); groups.append(group); texts.append(text)
        flags = review_flags(row["facts"])
        counts.update({f["kind"] for f in flags})
        reviews.append({"input": name, "record_index": i, "id": row["id"], "split": row["split"],
                        "facts": row["facts"], "charge": row["charge"], "sentence_months": row.get("sentence_months"),
                        "source_id": row.get("source_id"), "flags": flags, "overlap_pair_indices": [],
                        "review_decision": None, "notes": None})
    for n, pair in enumerate(pairs):
        for i in (pair["left_index"], pair["right_index"]):
            reviews[i]["overlap_pair_indices"].append(n)
    flagged = [r for r in reviews if r["flags"] or r["overlap_pair_indices"]]
    focus_ids = {r["id"] for r in focus} if focus is not None else None
    indexed = defaultdict(list)
    for _, row in entries:
        indexed[row["id"]].append(row)
    if focus is not None:
        if any(not any(row == other for other in indexed[row["id"]]) for row in focus):
            raise ValueError("Focus rows must be unchanged members of supplied inputs")
    report = {"version": VERSION, "input_hashes": {name: digest(rows) for name, rows in named_rows.items()},
              "input_rows": {name: len(rows) for name, rows in named_rows.items()}, "total_rows": len(entries),
              "flagged_rows": len(flagged), "pattern_flagged_rows": sum(bool(r["flags"]) for r in reviews),
              "risk_counts_rows_not_spans": dict(counts), "near_overlap_threshold": threshold,
              "overlap_pairs": len(pairs), "cross_split_pairs": sum(p["cross_split"] for p in pairs),
              "overlap_rows": sum(bool(r["overlap_pair_indices"]) for r in reviews),
              "labels_changed": False, "facts_changed": False, "splits_changed": False,
              "api_calls": 0, "training_ready": False,
              "limitations": ["Regex and lexical overlap are review candidates, not confirmed leakage or duplicate labels",
                              "Prior convictions may trigger sentence patterns; retain useful facts after review",
                              "No automatic row deletion, clause stripping, relabeling, or resplitting",
                              "No flag does not guarantee clean input or matching target defendant/charge"]}
    if focus_ids is not None:
        report["focus"] = {"rows": len(focus), "input_hash": digest(focus),
                           "flagged_rows": sum(r["id"] in focus_ids for r in flagged),
                           "flagged_ids": sorted({r["id"] for r in flagged if r["id"] in focus_ids})}
    return report, reviews, pairs
