"""Build a labeled training pool from an official CAIL2018 train member."""
import collections
import io
import json
import random
import re
import zipfile
from pathlib import Path

from .data import canonicalize, fact_hash, normalized_fact, split_train, visible_case
from .io import digest, read_rows, write_json, write_rows


def charge_key(value):
    value = value.strip().strip("[]【】")
    return re.sub(r"[\[\]【】，,、\s]", "", value.removesuffix("罪"))


def adapt(raw, source_id, charge_map):
    """Retain original numeric labels; reject unsupported prediction targets."""
    meta = raw.get("meta", {})
    charges = meta.get("accusation", [])
    if not isinstance(charges, list) or len(charges) != 1:
        return None, "multiple_or_missing_charges"
    charge = charge_map.get(charge_key(charges[0]))
    if not charge:
        return None, "outside_target_charges"
    if not isinstance(meta.get("criminals"), list) or len(meta["criminals"]) != 1:
        return None, "multiple_or_missing_defendants"
    term = meta.get("term_of_imprisonment", {})
    if term.get("death_penalty") is not False or term.get("life_imprisonment") is not False:
        return None, "non_fixed_or_missing_sentence_type"
    months = term.get("imprisonment")
    if type(months) is not int or months <= 0:
        return None, "nonpositive_or_invalid_months"
    facts = raw.get("fact")
    if not isinstance(facts, str) or not facts.strip():
        return None, "missing_facts"
    # Conservative initial pool: also excludes some legitimate prior-conviction descriptions.
    if re.search(r"判处.{0,20}(有期徒刑|拘役|无期徒刑|死刑)|判决如下|建议判处|量刑建议", facts):
        return None, "possible_outcome_text_review"
    row = canonicalize([{"source_id": source_id, "facts": facts, "charge": charge,
                          "sentence_months": months}], "cail2018_small", "train")[0]
    row["label_provenance"] = {"sentence_months": "original_dataset_meta.term_of_imprisonment.imprisonment",
                                "charge": "original_dataset_meta.accusation", "opinion": "missing"}
    row["relevant_articles_label"] = meta.get("relevant_articles", [])
    return row, None


def grams(text):
    text = normalized_fact(text)
    return {text[i:i+5] for i in range(max(1, len(text)-4))}


class BenchmarkGuard:
    """Exact checks plus a lexical near-overlap screen, not a semantic deduplicator."""
    def __init__(self, rows, threshold=.85):
        self.threshold = threshold
        self.rows = rows
        self.sets = [grams(r["facts"]) for r in rows]
        self.exact = {normalized_fact(r["facts"]): r["id"] for r in rows}
        self.postings = collections.defaultdict(set)
        for i, group in enumerate(self.sets):
            for gram in group:
                self.postings[gram].add(i)

    def match(self, facts):
        exact = self.exact.get(normalized_fact(facts))
        if exact:
            return {"benchmark_id": exact, "method": "normalized_exact", "score": 1.0}
        group = grams(facts)
        shared = collections.Counter(i for gram in group for i in self.postings.get(gram, ()))
        # All candidates considered; no top-k truncation that could hide a high-overlap pair.
        for i, common in shared.items():
            other = self.sets[i]
            denominator = min(len(group), len(other))
            if denominator < 20:
                continue
            containment = common / denominator
            if containment >= self.threshold:
                return {"benchmark_id": self.rows[i]["id"], "method": "character_5gram_containment",
                        "score": round(containment, 4)}
        return None


def build(archive, member, benchmarks, output_dir, per_charge=1000, seed=42):
    if per_charge <= 0:
        raise ValueError("per_charge must be positive")
    split_tokens = re.split(r"[/_.-]+", member.lower())
    if "train" not in split_tokens or any(s in split_tokens for s in ("test", "valid", "dev")):
        raise ValueError("Choose an actual original TRAIN member, never a held-out split")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new pool output directory")
    held_out = [r for path in benchmarks for r in read_rows(path)]
    charge_map = {charge_key(r["charge"]): r["charge"] for r in held_out}
    guard = BenchmarkGuard(held_out)
    counts, groups = collections.Counter(), collections.defaultdict(list)
    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(member)
        with bundle.open(info) as stream:
            for n, line in enumerate(io.TextIOWrapper(stream, encoding="utf-8-sig"), 1):
                if not line.strip():
                    continue
                counts["raw_rows"] += 1
                row, reason = adapt(json.loads(line), f"{member}:{n}", charge_map)
                if reason:
                    counts[reason] += 1
                else:
                    groups[fact_hash(row)].append(row)
        member_info = {"name": member, "crc32": f"{info.CRC:08x}", "uncompressed_bytes": info.file_size}
    unique = []
    for group in groups.values():
        if len({(r["charge"], r["sentence_months"]) for r in group}) != 1:
            counts["conflicting_duplicate_label_rows"] += len(group)
            continue
        unique.append(min(group, key=lambda r: r["source_id"]))
        counts["duplicate_rows_removed"] += len(group) - 1
    rng = random.Random(seed)
    rng.shuffle(unique)
    buckets, rejected = collections.defaultdict(list), []
    for row in unique:
        if len(buckets[row["charge"]]) >= per_charge:
            continue
        overlap = guard.match(row["facts"])
        if overlap:
            counts["benchmark_overlap_removed"] += 1
            rejected.append({"id": row["id"], **overlap})
            continue
        buckets[row["charge"]].append(row)
    pool = sorted([r for group in buckets.values() for r in group], key=lambda r: r["id"])
    train, dev = split_train(pool, .1, seed)
    dev_guard = BenchmarkGuard(dev)
    train_dev_rejected = []
    clean_train = []
    for row in train:
        overlap = dev_guard.match(row["facts"])
        if overlap:
            train_dev_rejected.append({"id": row["id"], **overlap})
        else:
            clean_train.append(row)
    train = clean_train
    if not train:
        raise ValueError("No training cases remain after train/dev lexical overlap screening")
    counts["train_dev_near_overlap_removed"] = len(train_dev_rejected)
    write_rows(out / "train.jsonl", train)
    write_rows(out / "dev.jsonl", dev)
    manifest = {"source_url": "https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip",
                "archive_bytes": Path(archive).stat().st_size, "member": member_info, "seed": seed,
                "per_charge_cap": per_charge, "counts": dict(counts),
                "train": len(train), "dev": len(dev), "pool_hash": digest(pool),
                "train_hash": digest(train), "dev_hash": digest(dev),
                "charges": {c: len(v) for c, v in sorted(buckets.items())},
                "benchmark_input_hash": digest([visible_case(r) for r in held_out]),
                "near_overlap_threshold": guard.threshold,
                "limitations": ["Lexical screen does not rule out semantic/case-level overlap",
                                 "Conservative outcome-text filter may remove prior-conviction cases",
                                 "CAIL is no longer an independent-source OOD benchmark when trained on CAIL",
                                 "No court reasoning available; teacher reasoning must be labeled synthetic"]}
    write_json(out / "manifest.json", manifest)
    write_json(out / "benchmark_overlap_removed.json", rejected)
    write_json(out / "train_dev_overlap_removed.json", train_dev_rejected)
    return manifest
