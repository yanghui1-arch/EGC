"""User-run E2 cohorts, bounded API matrix, and coverage-aware comparisons."""
from collections import Counter
import getpass
import os
from pathlib import Path
import random
import time

from . import facts
from .annotate import ApiRequestError, call_deepseek, decode_response
from .cail import BenchmarkGuard, input_risks
from .data import fact_hash, visible_case
from .io import digest, index_unique, read_json, read_rows, write_json, write_rows


def prepare(pool, exposed, regression, profile, output_dir, per_charge=30, seed=42):
    indexed = index_unique(pool)
    expectations = index_unique(regression["items"])
    if not pool or any(r["split"] != "train" for r in pool) or per_charge <= 0:
        raise ValueError("Use a nonempty original train pool and positive per-charge count")
    if not expectations or not set(expectations) <= indexed.keys():
        raise ValueError("Regression cases missing from pool; do not silently substitute cases")
    if any(input_risks(r["facts"]) for r in pool):
        raise ValueError("Pool still contains screened outcome/appeal inputs")
    condition_ids = {c["id"] for c in profile["conditions"]}
    for expected in expectations.values():
        if (not expected["expected"] or not set(expected["expected"]) <= condition_ids or
                not set(expected["expected"].values()) <= {"supported", "refuted", "unknown"}):
            raise ValueError("Invalid regression expectations")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh cohort directory to freeze selection")
    old = [indexed[i] for i in expectations]
    seen = exposed + old
    seen_ids = {r["id"] for r in seen}
    seen_sources = {(r["dataset"], r["source_id"]) for r in seen}
    guard = BenchmarkGuard(seen)
    rng = random.Random(seed)
    chosen, exclusions, selected_hashes = [], [], set()
    counts = {}
    for charge in ("诈骗罪", "抢劫罪"):
        candidates = sorted((r for r in pool if r["charge"] == charge), key=lambda r: r["id"])
        rng.shuffle(candidates)
        bucket = []
        for row in candidates:
            if len(bucket) == per_charge:
                break
            if row["id"] in seen_ids or (row["dataset"], row["source_id"]) in seen_sources:
                continue
            overlap = guard.match(row["facts"])
            if overlap:
                exclusions.append({"id": row["id"], **overlap})
                continue
            if fact_hash(row) in selected_hashes:
                continue
            selected_hashes.add(fact_hash(row))
            bucket.append(row)
        counts[charge] = len(bucket)
        chosen.extend(bucket)
    if not chosen:
        raise ValueError("No unexposed training cases remain")
    manifest = {"version": facts.VERSION, "seed": seed, "profile_hash": digest(profile),
        "pool_hash": digest(pool), "exposed_hash": digest(exposed),
        "regression_hash": digest(old), "new_hash": digest(chosen),
        "expectations_hash": digest(regression), "requested_per_charge": per_charge,
        "new_counts": counts, "regression_cases": len(old), "new_cases": len(chosen),
        "near_overlap_rejections": exclusions, "labels_unchanged": True,
        "limitation": "New cases exclude the supplied exposure files lexically; not a semantic independence guarantee."}
    write_rows(out / "regression.jsonl", old)
    write_rows(out / "new.jsonl", chosen)
    write_json(out / "expectations.json", regression)
    write_json(out / "profile.json", profile)
    write_json(out / "selection_manifest.json", manifest)
    return manifest


def _record(task, cache, row, profile):
    base = {"id": row["id"], "mode": task["mode"], "request_hash": task["hash"]}
    raw_path = cache / (task["hash"] + ".json")
    if raw_path.exists():
        saved = read_json(raw_path)
        if saved["request_hash"] != task["hash"]:
            raise ValueError("Cached request identity mismatch")
        base.update(usage=saved["envelope"].get("usage", {}), elapsed_seconds=saved.get("elapsed_seconds"))
        try:
            body, _, metadata = decode_response(saved["envelope"], True)
            result = facts.validate(body, row, profile, task["mode"])
            return {**base, "status": "accepted", "response": metadata, "result": result}
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            # No raw response fragments in the error log (JSONDecodeError contains location only).
            return {**base, "status": "rejected", "reason": str(exc)}
    error_path = cache / (task["hash"] + ".error.json")
    pending_path = cache / (task["hash"] + ".pending.json")
    if error_path.exists() or pending_path.exists():
        if error_path.exists():
            stored = read_json(error_path)
            if "diagnostic" in stored:
                diagnostic = stored["diagnostic"]
                base["diagnostic"] = ApiRequestError(diagnostic.get("category"), diagnostic.get("code")).diagnostic
        return {**base, "status": "api_error", "reason": "Previous request outcome uncertain; no automatic retry",
                "billing_outcome_may_be_unknown": True}
    return {**base, "status": "pending"}


def summarize(rows, records, modes, expectations=None):
    by_key = {(r["id"], r["mode"]): r for r in records}
    if len(by_key) != len(records) or set(by_key) != {(r["id"], m) for r in rows for m in modes}:
        raise ValueError("Comparison requires exactly one record for every requested case/mode")
    report = {"requested_cases": len(rows), "by_mode": {}, "paired": {}, "semantic_accuracy": None,
              "training_ready": False, "limitation": "Coverage and changes are not semantic correctness or benchmark gains."}
    for mode in modes:
        group = [r for r in records if r["mode"] == mode]
        states = Counter(c["status"] for r in group if r["status"] == "accepted"
                         for c in r["result"]["annotation"]["conditions"])
        report["by_mode"][mode] = {"records": dict(Counter(r["status"] for r in group)),
            "condition_states": dict(states),
            "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in group),
            "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in group),
            "measured_seconds": sum(r.get("elapsed_seconds") or 0 for r in group)}
    comparisons = [(a, b) for a, b in (("joint", "flat"), ("flat", "bound"), ("joint", "bound"), ("flat_v2", "bound_v2"))
                   if a in modes and b in modes]
    for left, right in comparisons:
        changes, valid_pairs = [], 0
        for row in rows:
            a, b = by_key[row["id"], left], by_key[row["id"], right]
            if a["status"] != "accepted" or b["status"] != "accepted":
                continue
            valid_pairs += 1
            a_states = {c["condition_id"]: c["status"] for c in a["result"]["annotation"]["conditions"]}
            b_states = {c["condition_id"]: c["status"] for c in b["result"]["annotation"]["conditions"]}
            for cid in a_states:
                if a_states[cid] != b_states[cid]:
                    changes.append({"id": row["id"], "condition_id": cid,
                                    "before": a_states[cid], "after": b_states[cid]})
        report["paired"][left + "_vs_" + right] = {"valid_pairs": valid_pairs,
            "not_comparable_cases": len(rows) - valid_pairs, "state_changes": changes}
    if expectations is not None:
        wanted = index_unique(expectations["items"])
        if not wanted.keys() <= {r["id"] for r in rows}:
            raise ValueError("Regression expectations reference cases outside this run")
        agreement = {}
        for mode in modes:
            items = []
            for identifier, item in wanted.items():
                rec = by_key[identifier, mode]
                states = ({c["condition_id"]: c["status"] for c in rec["result"]["annotation"]["conditions"]}
                          if rec["status"] == "accepted" else {})
                for cid, expected in item["expected"].items():
                    items.append({"id": identifier, "condition_id": cid, "expected": expected,
                                  "observed": states.get(cid), "match": states.get(cid) == expected})
            agreement[mode] = {"matched": sum(x["match"] for x in items), "total": len(items), "items": items}
        report["regression_agreement_not_accuracy"] = agreement
        report["expectation_kind"] = expectations.get("label_kind", "unspecified_not_gold")
        report["expectations_hash"] = digest(expectations)
    return report


def run(rows, profile, output_dir, modes=facts.MODES, model="deepseek-flash", limit=18,
        max_tokens=4096, ask_key=False, dry_run=False):
    index_unique(rows)
    index_unique(profile["conditions"])
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("E2 development runs accept train only, never dev/test")
    if any(input_risks(r["facts"]) for r in rows):
        raise ValueError("Screen outcome/appeal inputs before E2")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(facts.ALL_MODES):
        raise ValueError("Choose unique supported fact modes")
    version = facts.version_for(modes)
    if limit <= 0 or max_tokens <= 0 or not model:
        raise ValueError("Invalid request bounds/model")
    tasks = []
    for row in rows:
        for mode in modes:
            request = facts.payload(row, profile, mode, model, max_tokens)
            tasks.append({"id": row["id"], "mode": mode, "request": request, "hash": digest([row["id"], request])})
    identity = digest({"version": version, "source_hash": digest(rows), "profile": profile,
                       "requests": [t["hash"] for t in tasks]})
    out = Path(output_dir)
    manifest_path, cache = out / "manifest.json", out / "raw"
    if manifest_path.exists():
        if read_json(manifest_path)["identity"] != identity:
            raise ValueError("Run identity changed; use a new output directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty run directory has no valid manifest")
    indexed = index_unique(rows)
    records = [_record(t, cache, indexed[t["id"]], profile) for t in tasks]
    remaining = sum(r["status"] == "pending" for r in records)
    if dry_run:
        return {"dry_run": True, "cases": len(rows), "modes": list(modes), "total_requests": len(tasks),
                "remaining_requests": remaining, "new_requests_this_run_at_most": min(limit, remaining),
                "max_completion_tokens_this_run": min(limit, remaining) * max_tokens,
                "api_calls": 0, "writes": 0}
    out.mkdir(parents=True, exist_ok=True)
    # Fail closed on concurrent runs. A killed process may leave this lock for explicit inspection.
    lock_path = out / ".run.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Run is locked; ensure no process is running before removing .run.lock") from None
    calls = 0
    key = None
    def snapshot():
        report = summarize(rows, records, modes)
        write_rows(out / "records.jsonl", records)
        write_json(out / "report.json", report)
        manifest = {"identity": identity, "version": version, "source_hash": digest(rows),
            "profile": profile, "model": model, "modes": list(modes), "max_tokens": max_tokens,
            "case_ids": [r["id"] for r in rows], "transport": "curl_no_retry",
            "statuses": dict(Counter(r["status"] for r in records)), "new_attempts_this_invocation": calls,
            "complete": all(r["status"] != "pending" for r in records),
            "all_structurally_valid": all(r["status"] == "accepted" for r in records),
            "reference_labels_sent": False, "expectations_sent": False,
            "requests": [{k: t[k] for k in ("id", "mode", "hash")} for t in tasks]}
        if version != facts.VERSION:
            manifest["pending_requests"] = sum(r["status"] == "pending" for r in records)
            manifest["failed_requests"] = sum(r["status"] in {"api_error", "rejected"} for r in records)
            manifest["complete_meaning"] = "All tasks have a terminal status; not all requests succeeded"
        write_json(manifest_path, manifest)
        return manifest
    try:
        # Another process could have finished between the preflight read and acquiring the lock.
        if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
            raise ValueError("Run identity changed while acquiring lock")
        records[:] = [_record(t, cache, indexed[t["id"]], profile) for t in tasks]
        snapshot()
        for i, task in enumerate(tasks):
            if records[i]["status"] != "pending" or calls >= limit:
                continue
            if key is None:
                key = os.environ.get("DEEPSEEK_API_KEY")
                if not key and ask_key:
                    key = getpass.getpass("DeepSeek API Key (not saved): ")
                if not key:
                    raise ValueError("Use --ask-key or set DEEPSEEK_API_KEY")
            write_json(cache / (task["hash"] + ".pending.json"), {"request_hash": task["hash"], "id": task["id"]})
            calls += 1
            started = time.perf_counter()
            def capture(envelope):
                write_json(cache / (task["hash"] + ".json"), {"request_hash": task["hash"],
                    "elapsed_seconds": time.perf_counter() - started, "envelope": envelope})
            stop = False
            try:
                call_deepseek(task["request"], key, True, "curl", capture_response=capture)
            except (RuntimeError, ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                if not (cache / (task["hash"] + ".json")).exists():
                    error = {"error_type": type(exc).__name__, "outcome": "unknown_no_automatic_retry"}
                    if isinstance(exc, ApiRequestError):
                        error["diagnostic"] = exc.diagnostic
                        print(str(exc), flush=True)
                    write_json(cache / (task["hash"] + ".error.json"), error)
                    stop = True
            records[i] = _record(task, cache, indexed[task["id"]], profile)
            snapshot()
            print(f"Requests attempted this run {calls}/{limit}; {task['mode']}: {records[i]['status']}", flush=True)
            if stop:
                break
        return snapshot()
    finally:
        lock.close()
        lock_path.unlink()


def report_run(rows, run_dir, output, expectations=None):
    directory = Path(run_dir)
    manifest = read_json(directory / "manifest.json")
    if digest(rows) != manifest["source_hash"]:
        raise ValueError("Report source differs from frozen run")
    records = read_rows(directory / "records.jsonl")
    # Rebuild validation from immutable raw responses instead of trusting edited derived states.
    tasks = [{"id": t["id"], "mode": t["mode"], "hash": t["hash"]} for t in manifest["requests"]]
    indexed = index_unique(rows)
    rebuilt = [_record(t, directory / "raw", indexed[t["id"]], manifest["profile"]) for t in tasks]
    if records != rebuilt:
        raise ValueError("Derived records differ from raw responses; resume run offline from cache to refresh")
    report = summarize(rows, records, manifest["modes"], expectations)
    report.update(run_identity=manifest["identity"], source_hash=manifest["source_hash"],
                  version=manifest["version"], profile_hash=digest(manifest["profile"]))
    if any(mode.endswith("_v2") for mode in manifest["modes"]):
        # All conditions, including unchanged/unknown states and failures, must be reviewed.
        queue = []
        by_key = {(r["id"], r["mode"]): r for r in records}
        for row in rows:
            for condition in manifest["profile"]["conditions"]:
                predictions = {}
                for mode in manifest["modes"]:
                    record = by_key[row["id"], mode]
                    prediction = {"record_status": record["status"], "state": None, "evidence": []}
                    if record["status"] == "accepted":
                        c = next(c for c in record["result"]["annotation"]["conditions"] if c["condition_id"] == condition["id"])
                        prediction.update(state=c["status"], evidence=c["evidence"])
                    predictions[mode] = prediction
                queue.append({"id": row["id"], "condition": condition, "facts": row["facts"],
                              "predictions": predictions, "review_state": None, "review_notes": None,
                              "review_kind": "unreviewed_not_gold", "run_identity": manifest["identity"]})
        queue_path = Path(output).with_name("review_queue.jsonl")
        if queue_path == Path(output):
            raise ValueError("Report output cannot be named review_queue.jsonl")
        write_rows(queue_path, queue)
        report.update(review_queue=str(queue_path), review_items=len(queue), review_complete=False)
    write_json(output, report)
    return report
