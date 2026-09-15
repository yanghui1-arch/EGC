"""CPU-only conservative screening; preserve rows/splits and quarantine intact inputs."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egc.cail import INPUT_SCREEN_VERSION, input_risks
from egc.io import digest, index_unique, read_rows, write_json, write_rows


def screen(paths, output):
    rows = [row for path in paths for row in read_rows(path)]
    index_unique(rows)
    if any(row["split"] not in {"train", "dev"} for row in rows):
        raise ValueError("Only existing train/dev rows may be screened")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh screening output directory")
    kept, quarantine = [], []
    for row in rows:
        flags = input_risks(row["facts"])
        if flags:
            quarantine.append({"case": row, "flags": flags})
        else:
            kept.append(row)
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in kept if r["split"] == split]
        if not selected:
            raise ValueError("Screening would empty a split")
    for split in sorted({r["split"] for r in rows}):
        write_rows(out / f"{split}.jsonl", [r for r in kept if r["split"] == split])
    write_rows(out / "quarantine.jsonl", quarantine)
    report = {"screen_version": INPUT_SCREEN_VERSION, "input_hash": digest(rows),
        "retained_hash": digest(kept), "input_cases": len(rows), "retained_cases": len(kept),
        "quarantined_cases": len(quarantine), "splits": dict(Counter(r["split"] for r in kept)),
        "flags": dict(Counter(f for r in quarantine for f in r["flags"])),
        "labels_facts_and_splits_unchanged": True,
        "limitation": "Conservative lexical screen, not proof of no outcome information or semantic errors."}
    write_json(out / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(screen(args.input, args.output_dir))
