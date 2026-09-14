"""Package only label-free base jobs and manifests; never upload the archive to Git."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egc.io import digest, index_unique, read_json, read_rows


def pack(root, output):
    members = {}
    allowed_job_keys = {"id", "variant", "split", "messages", "prompt_hash"}
    for dataset in ("laic_test", "pccd_test", "cail_test"):
        directory = root / "data/prepared/base" / dataset
        jobs, manifest = read_rows(directory / "jobs.jsonl"), read_json(directory / "manifest.json")
        index_unique(jobs)
        if not jobs or manifest["jobs_hash"] != digest(jobs):
            raise ValueError("Empty or modified prepared jobs")
        for job in jobs:
            if set(job) != allowed_job_keys or job["variant"] != "base" or job["split"] != "test":
                raise ValueError("Only label-free base test jobs are allowed in this bundle")
            if job["prompt_hash"] != digest(job["messages"]):
                raise ValueError("Prompt fingerprint mismatch")
            payload = json.loads(job["messages"][-1]["content"])
            if set(payload) != {"facts", "charge"}:
                raise ValueError("Unexpected base prompt fields")
        for name in ("jobs.jsonl", "manifest.json"):
            file = directory / name
            members[file.relative_to(root).as_posix()] = file.read_bytes()
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in members.items()}
    output = Path(output)
    if output.exists():
        raise ValueError("Archive exists; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr("base_jobs_checksums.json", json.dumps(checksums, indent=2))
    print(json.dumps({"archive": str(output.resolve()), "files": len(members), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/base_jobs.zip")
    args = parser.parse_args()
    pack(ROOT, args.output)
