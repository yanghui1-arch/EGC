"""User-run local correction: implicit target inputs with label-only target mentions.

Preserves every case, fact, original month label, annotation, and held-out test byte.
Only train/dev prompts lose target_person; its literal fact spans move to label-only
sidecars ({split}.target_supervision.jsonl) for TERA v2 pointer supervision/diagnostics.
No API calls or test-label reweighting.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .b0 import encoded, lines, pack_files
from .io import digest, write_json
from .learned import ARMS, VERSION, fresh, messages
from .learned_audit import checked_zip
from .learned_server import validate_data
from .tera_data import target_char_spans

CORRECTION = "implicit-target-sentence-v2"
TOKEN_MEAN = "completion-token-mean-v1"
BALANCED = "sentence-rest-half-v1"
TARGET_SUPERVISION = "target-name-char-spans-v1"


def sentence_policy(manifest):
    correction = manifest.get("training_correction")
    if correction is None:
        return TOKEN_MEAN
    if (correction.get("version") != CORRECTION or correction.get("input_target") != "absent"
            or correction.get("sentence_loss") not in (TOKEN_MEAN, BALANCED)
            or correction.get("target_supervision") != TARGET_SUPERVISION):
        raise ValueError("Unknown training correction policy")
    return correction["sentence_loss"]


def build(archive, output, balanced_sentence_loss=False):
    archive = Path(archive)
    files = checked_zip(archive)
    manifest = json.loads(files["manifest.json"])
    if manifest["version"] != VERSION or manifest["arms"] != list(ARMS) or "training_correction" in manifest:
        raise ValueError("Expected an uncorrected learned package; never recorrect a package")
    # Reuse the existing allowlist and canonical validation before creating any result.
    from tempfile import TemporaryDirectory
    from .learned_server import unpack
    with TemporaryDirectory() as temp:
        unpack(archive, Path(temp) / "source")
    changed, profiles = {}, {}
    for split in ("train", "dev"):
        rows = [json.loads(line) for line in files[f"{split}.references.jsonl"].decode().splitlines()]
        if not all(isinstance(r.get("target_person"), str) and r["target_person"].strip() for r in rows):
            raise ValueError("Expected explicit single-target training/dev source rows")
        corrected = [{k: v for k, v in r.items() if k != "target_person"} for r in rows]
        changed[f"{split}.references.jsonl"] = lines(corrected)
        mentions = [{"id": r["id"], "target_char_spans": target_char_spans(r["facts"], r["target_person"])} for r in rows]
        changed[f"{split}.target_supervision.jsonl"] = lines(mentions)
        profiles[split] = {"n": len(rows), "before_hash": digest(rows), "after_hash": digest(corrected),
                           "target_before": len(rows), "target_after": 0,
                           "cases_with_target_mentions": sum(bool(m["target_char_spans"]) for m in mentions),
                           "target_mentions": sum(len(m["target_char_spans"]) for m in mentions),
                           "month_counts": dict(sorted(Counter(r["sentence_months"] for r in rows).items()))}
        for arm in ARMS:
            name = f"{arm}.{split}.sft.jsonl"
            sft = [json.loads(line) for line in files[name].decode().splitlines()]
            if [r["id"] for r in sft] != [r["id"] for r in corrected]:
                raise ValueError("Unpaired source cases")
            changed[name] = lines([dict(item, prompt=messages(ref, arm)) for item, ref in zip(sft, corrected)])
            if split == "dev":
                name = f"{arm}.dev.jobs.jsonl"
                jobs = [json.loads(line) for line in files[name].decode().splitlines()]
                changed[name] = lines([dict(job, messages=messages(ref, arm), prompt_hash=digest(messages(ref, arm)))
                                       for job, ref in zip(jobs, corrected)])
    policy = {"version": CORRECTION, "input_target": "absent", "target_supervision": TARGET_SUPERVISION,
              "sentence_loss": BALANCED if balanced_sentence_loss else TOKEN_MEAN,
              "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "cases_resampled": False, "test_labels_used_for_weights": False}
    report = {**policy, "splits": profiles, "new_api_calls": 0,
              "tests_byte_identical": True, "completion_bytes_unchanged": True,
              "note": "Teacher roles and literal target-name spans are label-only supervision; no target field at "
                      "inference. Name spans are string matches, not verified coreference. Equal field loss is an "
                      "optional ablation, not proof of distribution calibration or module gain."}
    manifest["training_correction"] = policy
    changed["manifest.json"] = encoded(manifest)
    # Validate the corrected package with the same server checks before exporting it.
    with TemporaryDirectory() as temp:
        check = Path(temp)
        for name, content in (files | changed).items():
            (check / name).write_bytes(content)
        validate_data(check)
        from .tera_server import frozen_jobs
        for entry in manifest["tests"]:
            frozen_jobs(check, entry["name"])
    out = fresh(output)
    pack_files(out / "experiment.zip", files | changed)
    report["output_archive_sha256"] = hashlib.sha256((out / "experiment.zip").read_bytes()).hexdigest()
    write_json(out / "correction.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default="data/learned_v2_screened/experiment.zip")
    p.add_argument("--output", default="data/learned_v3_latent")
    p.add_argument("--balanced-sentence-loss", action="store_true",
                   help="Ablation: half of the SFT loss on the month digits (default keeps token-mean loss)")
    print(json.dumps(build(**vars(p.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
