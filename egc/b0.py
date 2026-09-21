"""B0 user-run CPU packaging/evaluation and server inference orchestration."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from .e4_inventory import source_metadata
from .evaluate import parse_output, summarize
from .io import digest, index_unique, read_json, read_rows, write_json, write_rows
from .prompts import SYSTEM

PROTOCOL = Path(__file__).resolve().parents[1] / "configs/b0_protocol.json"
ARMS = ("original", "target")
B0_SYSTEM = SYSTEM + (
    "若输入提供target_person，则只预测该人物在给定罪名下的刑期，"
    "不要将其他人物的情节归给该人物；该字段只指定对象，不证明任何量刑情节。"
)
INPUT_FILES = {"manifest.json", "target_mask.json", "original.jobs.jsonl", "target.jobs.jsonl"}
RESULT_FILES = {"manifest.json", "target_mask.json", "completion.json", "execution.json"} | {
    name for arm in ARMS for name in (f"predictions_{arm}.jsonl", f"predictions_{arm}.jsonl.manifest.json")
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def lines(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def decode_rows(data):
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty B0 rows")
    index_unique(rows)
    return rows


def pack_files(output, members):
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("checksums.json", encoded(checksums))


def read_archive(path, allowed):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != allowed | {"checksums.json"}:
            raise ValueError("Unexpected/duplicate archive members")
        if any(info.file_size > 20_000_000 for info in archive.infolist()):
            raise ValueError("Oversized B0 archive member")
        data = {name: archive.read(name) for name in names}
    expected = json.loads(data.pop("checksums.json"))
    if expected != {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}:
        raise ValueError("Archive checksum mismatch")
    return data


def build(rows, metadata, protocol):
    index_unique(rows)
    if any(r.get("split") != "dev" for r in rows):
        raise ValueError("B0 accepts only the frozen dev split")
    if digest(rows) != protocol["dev_hash"] or len(rows) != protocol["dev_rows"]:
        raise ValueError("Frozen dev snapshot mismatch")
    selected = sorted((r for r in rows if r["charge"] in protocol["charges"]), key=lambda r: r["id"])
    if dict(Counter(r["charge"] for r in selected)) != protocol["charges"]:
        raise ValueError("Frozen charge counts mismatch")
    jobs, masks, references = {arm: [] for arm in ARMS}, [], []
    for row in selected:
        meta = metadata[row["id"]]
        names = meta["target_names_from_meta"]
        available = len(names) == 1 and names[0] in row["facts"]
        masks.append({"id": row["id"], "available": available,
                      "reason": "single_source_name_literal_match" if available
                      else "source_name_missing_multiple_or_not_in_facts"})
        for arm in ARMS:
            body = {"facts": row["facts"], "charge": row["charge"]}
            if arm == "target" and available:
                body["target_person"] = names[0]
            messages = [{"role": "system", "content": B0_SYSTEM},
                        {"role": "user", "content": json.dumps(body, ensure_ascii=False, sort_keys=True)}]
            jobs[arm].append({"id": row["id"], "variant": f"b0_{arm}", "split": "dev",
                              "messages": messages, "prompt_hash": digest(messages)})
        months = row["sentence_months"]
        if type(months) is not int or months < 0:
            raise ValueError("Invalid numeric reference")
        references.append({"id": row["id"], "charge": row["charge"],
                           "sentence_months": months, "opinion": None})
    manifest = {"version": protocol["version"], "protocol": protocol,
                "n_cases": len(selected), "n_generations": len(selected)*2,
                "jobs_hash": {a: digest(jobs[a]) for a in ARMS},
                "target_mask_hash": digest(masks), "api_calls": 0, "training": False}
    return jobs, masks, references, manifest


def validate_inputs(data, protocol):
    manifest, masks = json.loads(data["manifest.json"]), json.loads(data["target_mask.json"])
    if manifest.get("protocol") != protocol or manifest.get("version") != protocol["version"]:
        raise ValueError("B0 protocol differs from this code/config")
    mask = index_unique(masks)
    jobs = {a: decode_rows(data[f"{a}.jobs.jsonl"]) for a in ARMS}
    if manifest["target_mask_hash"] != digest(masks):
        raise ValueError("Target mask hash mismatch")
    expected_n = sum(protocol["charges"].values())
    if manifest["n_cases"] != expected_n or manifest["n_generations"] != expected_n*2:
        raise ValueError("B0 case count mismatch")
    for arm in ARMS:
        if len(jobs[arm]) != expected_n or manifest["jobs_hash"][arm] != digest(jobs[arm]):
            raise ValueError("B0 jobs hash/count mismatch")
        if set(index_unique(jobs[arm])) != set(mask):
            raise ValueError("B0 mask/job IDs differ")
        for job in jobs[arm]:
            if set(job) != {"id", "variant", "split", "messages", "prompt_hash"}:
                raise ValueError("Unexpected job fields; references cannot enter inference")
            if job["variant"] != f"b0_{arm}" or job["split"] != "dev":
                raise ValueError("Wrong B0 arm/split")
            messages = job["messages"]
            if (len(messages) != 2 or messages[0] != {"role": "system", "content": B0_SYSTEM}
                    or set(messages[1]) != {"role", "content"} or messages[1]["role"] != "user"
                    or digest(messages) != job["prompt_hash"]):
                raise ValueError("Wrong B0 prompt")
            body = json.loads(messages[1]["content"])
            expected = {"facts", "charge"} | ({"target_person"} if arm == "target" and mask[job['id']]['available'] else set())
            if set(body) != expected:
                raise ValueError("Unexpected prompt fields")
            if "target_person" in body and (not isinstance(body["target_person"], str)
                    or not body["target_person"] or body["target_person"] not in body["facts"]):
                raise ValueError("Target person is not in facts")
        charges = Counter(json.loads(j["messages"][1]["content"])["charge"] for j in jobs[arm])
        if dict(charges) != protocol["charges"]:
            raise ValueError("B0 job charges differ")
    if [j['id'] for j in jobs['original']] != [j['id'] for j in jobs['target']]:
        raise ValueError("B0 paired order differs")
    for first, second in zip(jobs["original"], jobs["target"]):
        body = json.loads(second["messages"][1]["content"])
        body.pop("target_person", None)
        if body != json.loads(first["messages"][1]["content"]):
            raise ValueError("Paired facts/charge differ")
    return manifest, masks, jobs


def prepare(dev, source_zip, output_dir, protocol):
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Use a fresh B0 output directory")
    rows = read_rows(dev)
    if any(r.get('split') != 'dev' for r in rows) or digest(rows) != protocol['dev_hash']:
        raise ValueError("Frozen dev snapshot mismatch")
    selected = [r for r in rows if r['charge'] in protocol['charges']]
    metadata = source_metadata(selected, source_zip)
    jobs, masks, references, manifest = build(rows, metadata, protocol)
    members = {f'{a}.jobs.jsonl': lines(jobs[a]) for a in ARMS}
    members.update({'manifest.json': encoded(manifest), 'target_mask.json': encoded(masks)})
    validate_inputs(members, protocol)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in members.items():
        (output / name).write_bytes(data)
    write_rows(output/'references.local.jsonl', references)
    write_json(output/'source_provenance.local.json', metadata)
    write_json(output/'local_manifest.json', {'references_hash': digest(references),
               'manifest_hash': digest(manifest), 'source_metadata_hash': digest(metadata)})
    pack_files(output/'b0_jobs.zip', members)
    return {'archive': str((output/'b0_jobs.zip').resolve()), 'cases': len(references),
            'generations': len(references)*2, 'target_available': sum(m['available'] for m in masks),
            'api_calls': 0, 'training': False,
            'archive_sha256': hashlib.sha256((output/'b0_jobs.zip').read_bytes()).hexdigest()}


def checked_predictions(data, jobs, protocol):
    predictions, manifests = {}, {}
    for arm in ARMS:
        rows = decode_rows(data[f'predictions_{arm}.jsonl'])
        lookup = index_unique(rows)
        manifest = json.loads(data[f'predictions_{arm}.jsonl.manifest.json'])
        settings = manifest['settings']
        if set(lookup) != {j['id'] for j in jobs[arm]}:
            raise ValueError("Missing/extra generation IDs")
        if (settings['jobs_hash'] != digest(jobs[arm]) or settings['model'] != protocol['model']
                or settings.get('adapter') is not None or digest(settings) != manifest['generation_key']):
            raise ValueError("Wrong inference identity/model")
        for key, value in protocol['decoding'].items():
            if key != 'gpu_memory' and settings.get(key) != value:
                raise ValueError(f"Decoding setting mismatch: {key}")
        for job in jobs[arm]:
            pred = lookup[job['id']]
            if pred.get('prompt_hash') != job['prompt_hash'] or pred.get('generation_key') != manifest['generation_key']:
                raise ValueError("Prediction does not match job/run")
        predictions[arm], manifests[arm] = rows, manifest
    for key in ('model_config', 'weight_inventory', 'adapter_config'):
        if manifests['original']['settings'].get(key) != manifests['target']['settings'].get(key):
            raise ValueError("Paired model weights/config differ")
    return predictions


def collect(input_data, run_dir, protocol):
    run = Path(run_dir)
    manifest, masks, jobs = validate_inputs(input_data, protocol)
    data = {name: (run/name).read_bytes() for name in RESULT_FILES - {'manifest.json', 'target_mask.json', 'completion.json'}}
    predictions = checked_predictions(data, jobs, protocol)
    completion = {'generation_complete': True, 'n_cases': manifest['n_cases'],
                  'valid_format': {arm: sum(parse_output(r['text']) is not None and r.get('finish_reason') != 'length'
                                             for r in predictions[arm]) for arm in ARMS},
                  'note': 'Generation completion is not format/semantic accuracy or a benchmark score.'}
    write_json(run/'completion.json', completion)
    data.update({'manifest.json': encoded(manifest), 'target_mask.json': encoded(masks), 'completion.json': encoded(completion)})
    pack_files(run/'b0_results.zip', data)
    return {'results_zip': str((run/'b0_results.zip').resolve()), **completion}


def server_run(archive, run_dir, protocol):
    # Called by the user on Linux; each arm uses a separate child process to release GPU memory.
    if not sys.platform.startswith('linux'):
        raise ValueError("B0 inference runs only on the user's Linux server")
    data = read_archive(archive, INPUT_FILES)
    validate_inputs(data, protocol)
    run = Path(run_dir)
    if run.exists():
        raise ValueError("Use a fresh server run directory; preserve failed runs")
    run.mkdir(parents=True, exist_ok=False)
    inputs = run/'input'; inputs.mkdir()
    for name, content in data.items():
        (inputs/name).write_bytes(content)
    execution = {'started_unix': time.time(), 'arms': {}, 'protocol_hash': digest(protocol)}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    execution['code_commit'] = revision.stdout.strip() if revision.returncode == 0 else None
    write_json(run/'execution.json', execution)
    for arm in ARMS:
        command = [sys.executable, '-u', '-m', 'egc', 'infer', '--model', protocol['model'],
                   '--jobs', str(inputs/f'{arm}.jobs.jsonl'), '--output', str(run/f'predictions_{arm}.jsonl')]
        for key, value in protocol['decoding'].items():
            command += ['--'+key.replace('_','-'), str(value)]
        start = time.monotonic()
        result = subprocess.run(command, check=False)
        execution['arms'][arm] = {'exit_code': result.returncode, 'wall_seconds_including_model_load': time.monotonic()-start}
        write_json(run/'execution.json', execution)
        if result.returncode:
            raise RuntimeError(f"B0 {arm} failed; preserve {run} and return run.log/execution.json")
    return collect(data, run, protocol)


def evaluate(prepared_dir, archive, output, protocol):
    root = Path(prepared_dir)
    if Path(output).exists():
        raise ValueError("Use a new metrics file")
    inputs = {name: (root/name).read_bytes() for name in INPUT_FILES}
    manifest, masks, jobs = validate_inputs(inputs, protocol)
    data = read_archive(archive, RESULT_FILES)
    if json.loads(data['manifest.json']) != manifest or json.loads(data['target_mask.json']) != masks:
        raise ValueError("Results belong to a different prepared package")
    references = read_rows(root/'references.local.jsonl')
    local = read_json(root/'local_manifest.json')
    if digest(references) != local['references_hash'] or digest(manifest) != local['manifest_hash']:
        raise ValueError("Local reference/package hash mismatch")
    if set(index_unique(references)) != {m['id'] for m in masks}:
        raise ValueError("Reference IDs differ")
    predictions = checked_predictions(data, jobs, protocol)
    available = {m['id'] for m in masks if m['available']}
    scopes = {'all_cases': references, 'target_available': [r for r in references if r['id'] in available]}
    scopes.update({f'charge:{charge}': [r for r in references if r['charge']==charge] for charge in protocol['charges']})
    report = {'protocol': protocol, 'target_available': len(available), 'scopes': {},
              'execution': json.loads(data['execution.json']),
              'interpretation': 'Dev input-information diagnostic only; not EGC/H1/H2 gain. No independent reasoning gold.'}
    for scope, refs in scopes.items():
        ids = {r['id'] for r in refs}
        scores = {arm: summarize(refs, [p for p in predictions[arm] if p['id'] in ids]) for arm in ARMS} if refs else None
        delta = None
        if scores and all(scores[a]['eligible_for_full_mae_comparison'] for a in ARMS):
            delta = scores['target']['mae_months_full'] - scores['original']['mae_months_full']
        report['scopes'][scope] = {'n': len(refs), 'arms': scores, 'delta_mae_target_minus_original': delta}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    q = subs.add_parser('prepare'); q.add_argument('--dev', required=True); q.add_argument('--source-zip', required=True); q.add_argument('--output-dir', required=True)
    q = subs.add_parser('run'); q.add_argument('--archive', required=True); q.add_argument('--run-dir', required=True)
    q = subs.add_parser('evaluate'); q.add_argument('--prepared-dir', required=True); q.add_argument('--results-zip', required=True); q.add_argument('--output', required=True)
    args = parser.parse_args(); protocol = read_json(PROTOCOL)
    if args.command == 'prepare': result = prepare(args.dev, args.source_zip, args.output_dir, protocol)
    elif args.command == 'run': result = server_run(args.archive, args.run_dir, protocol)
    else: result = evaluate(args.prepared_dir, args.results_zip, args.output, protocol)
    if args.command == 'evaluate':
        result = {'metrics_file': str(Path(args.output).resolve()), 'n': result['scopes']['all_cases']['n']}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
