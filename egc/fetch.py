import hashlib
import urllib.error
import urllib.request
from pathlib import Path

from .io import read_json, write_json


def fetch_upstream(manifest_path, destination):
    manifest = read_json(manifest_path)
    root = Path(destination).resolve()
    records = []
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe manifest path")
        url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}/{entry['path']}"
        expected = entry["git_blob_sha1"]
        def blob_hash(data):
            return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        data = path.read_bytes() if path.exists() else None
        if data is None or blob_hash(data) != expected:
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    data = response.read()
            except urllib.error.URLError:
                raise RuntimeError(f"Download failed for {entry['path']}; check GitHub network access") from None
            if len(data) != entry["bytes"] or blob_hash(data) != expected:
                raise ValueError(f"Upstream content hash mismatch: {entry['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        records.append({**entry, "sha256": hashlib.sha256(data).hexdigest()})
    result = {"repository": manifest["repository"], "revision": manifest["revision"], "files": records}
    write_json(root / "download_manifest.json", result)
    return result
