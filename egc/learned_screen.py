"""User-run, zero-API conservative input-risk quarantine for a prepared package.

Matches are review candidates, NOT confirmed leakage or legal judgments. Whole
cases are withheld from all arms; facts, labels, annotations and tests stay intact.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from .b0 import encoded, lines, pack_files
from .io import digest, index_unique, write_json, write_rows
from .learned import ARMS, VERSION, fresh

POLICY = "numeric-term-context-quarantine-v1"
TERM = re.compile(r"判处|量刑|判决如下|缓刑")
NUMBER = re.compile(r"[一二三四五六七八九十百两0-9]+[个年月]")


def risk_spans(facts):
    # Deliberately broad: dates, prior convictions and legal quotations can match.
    # The output must never be represented as a semantic contamination label.
    return [s for s in re.split(r"[。；\n]", facts) if TERM.search(s) and NUMBER.search(s)]


def build_screened(archive, output):
    archive = Path(archive)
    source_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any("/" in n or "\\" in n or n.startswith(".") for n in names):
            raise ValueError("Unsafe or duplicate package member")
        if sum(i.file_size for i in z.infolist()) > 512_000_000:
            raise ValueError("Oversized package")
        files = {n: z.read(n) for n in names if n != "checksums.json"}
        if json.loads(z.read("checksums.json")) != {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}:
            raise ValueError("Source package checksum mismatch")
    manifest = json.loads(files["manifest.json"])
    if manifest["version"] != VERSION or manifest["arms"] != list(ARMS) or "screening" in manifest:
        raise ValueError("Expected an unscreened context-v2 package")
    allowed = {"manifest.json", "train.references.jsonl", "dev.references.jsonl"}
    allowed |= {f"{a}.{s}.sft.jsonl" for a in ARMS for s in ("train", "dev")}
    allowed |= {f"{a}.dev.jobs.jsonl" for a in ARMS}
    for t in manifest["tests"]:
        allowed |= {f"{t['name']}.references.jsonl"} | {f"{a}.{t['name']}.jobs.jsonl" for a in ARMS}
    if set(files) != allowed:
        raise ValueError("Unexpected source members")
    def read(name):
        rows = [json.loads(s) for s in files[name].decode("utf-8").splitlines() if s.strip()]
        index_unique(rows)
        return rows
    quarantined = []
    previous_retained = dict(manifest["retained"])
    for split in ("train", "dev"):
        refs = read(f"{split}.references.jsonl")
        if len(refs) != previous_retained[split] or any(r["split"] != split for r in refs):
            raise ValueError("Source split/count mismatch")
        rejected = set()
        for r in refs:
            hits = risk_spans(r["facts"])
            if hits:
                rejected.add(r["id"])
                quarantined.append({"id": r["id"], "split": split, "input_hash": digest(r["facts"]),
                                    "matched_spans": hits, "semantic_verdict": "not_adjudicated"})
        retained = [r for r in refs if r["id"] not in rejected]
        if len(retained) < len(refs)*.5 or {r["charge"] for r in retained} != {r["charge"] for r in refs}:
            raise ValueError("Screening removed too many cases or an entire charge")
        for arm in ARMS:
            members = [f"{arm}.{split}.sft.jsonl"]
            if split == "dev":
                members.append(f"{arm}.dev.jobs.jsonl")
            for name in members:
                items = read(name)
                if [r["id"] for r in items] != [r["id"] for r in refs]:
                    raise ValueError("Source arms are not paired")
                files[name] = lines([r for r in items if r["id"] not in rejected])
        files[f"{split}.references.jsonl"] = lines(retained)
        manifest["retained"][split] = len(retained)
        manifest["retained_by_charge"][split] = dict(Counter(r["charge"] for r in retained))
    report = {"policy": POLICY, "source_archive_sha256": source_sha,
              "previous_retained": previous_retained, "retained": manifest["retained"],
              "quarantined": dict(Counter(r["split"] for r in quarantined)),
              "quarantine_manifest_hash": digest(quarantined),
              "semantic_leakage_accuracy": None, "api_calls": 0,
              "limitations": ["Matches can include prior convictions, dates and legal references",
                              "Unmatched cases are not certified free of leakage or reasoning contamination",
                              "Conservative selection changes the cohort; compare arms only within this snapshot",
                              "All test files are unchanged and unused for screening"]}
    manifest["annotation_rejected"] = manifest["rejected"]
    manifest["rejected"] += len(quarantined)
    manifest["screening"] = report
    files["manifest.json"] = encoded(manifest)
    out = fresh(output)
    pack_files(out / "experiment.zip", files)
    write_json(out / "manifest.json", manifest)
    write_json(out / "screening.json", report)
    write_rows(out / "quarantine.jsonl", quarantined)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/learned_v2/ready/experiment.zip")
    parser.add_argument("--output", default="data/learned_v2_screened")
    args = parser.parse_args()
    print(json.dumps(build_screened(args.archive, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
