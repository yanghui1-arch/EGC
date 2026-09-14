from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Empty data file: {path}")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSON objects: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def index_unique(rows, key="id"):
    result = {}
    for row in rows:
        identifier = row[key]
        if identifier in result:
            raise ValueError(f"Duplicate {key}: {identifier}")
        result[identifier] = row
    return result
