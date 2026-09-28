"""TERA v1: prompt-only spans and sparse train-only role supervision (no API)."""
import json
import re
from collections import Counter, defaultdict

from .server import render_chat, tokenize_sft_row

VERSION = "tera-sentence-memory-v1"
ROLES = ("target", "other", "uncertain")


def sentence_spans(text):
    # Syntax-only boundaries; no legal keyword classification. Long spans are chunked.
    cuts = [0] + [m.end() for m in re.finditer(r"[。！？；\n]+", text)] + [len(text)]
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        for start in range(a, b, 160):
            end = min(start + 160, b)
            if text[start:end].strip():
                spans.append((start, end))
    if not spans:
        raise ValueError("Empty fact memory")
    return spans


def weak_roles(facts, spans, evidence):
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Missing training evidence")
    votes = [set() for _ in spans]
    counts = Counter()
    for item in evidence:
        quote, relation = item["quote"], item["relation"]
        if not isinstance(quote, str) or not quote or relation not in ROLES:
            raise ValueError("Invalid weak label; never fill missing fields")
        start = facts.find(quote)
        if start < 0:
            raise ValueError("Non-source training quote")
        if facts.find(quote, start + 1) >= 0:
            counts["repeated_quote"] += 1
            continue
        located = [i for i, (a, b) in enumerate(spans) if a <= start and start + len(quote) <= b]
        if len(located) != 1:
            counts["cross_boundary_quote"] += 1
            continue
        votes[located[0]].add(ROLES.index(relation))
        counts["aligned_quotes"] += 1
    labels = []
    for vote in votes:
        label = next(iter(vote)) if len(vote) == 1 else -100
        labels.append(label)
        counts[ROLES[label] if label >= 0 else "conflict" if vote else "unlabelled"] += 1
    return labels, dict(counts)


def prompt_features(tokenizer, messages):
    """Offsets come from the serialized user JSON, never from an answer search."""
    content = messages[-1]["content"]
    visible = json.loads(content)
    if messages[-1]["role"] != "user" or set(visible) - {"facts", "charge", "target_person"}:
        raise ValueError("Unexpected inference fields")
    if content != json.dumps(visible, ensure_ascii=False, sort_keys=True):
        raise ValueError("Expected canonical existing direct prompt")
    for field in ("facts", "charge"):
        if not isinstance(visible.get(field), str) or not visible[field].strip():
            raise ValueError(f"Missing {field}")
    if "target_person" in visible and (not isinstance(visible["target_person"], str) or not visible["target_person"].strip()):
        raise ValueError("Invalid explicit target")
    rendered = render_chat(tokenizer, messages)
    if rendered.count(content) != 1:
        raise ValueError("Ambiguous serialized user boundary")
    origin = rendered.index(content)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if ids != tokenizer.encode(rendered, add_special_tokens=False) or not ids:
        raise ValueError("Tokenizer offset/ID mismatch")
    fields, cursor = {}, 1  # opening JSON object
    for key in sorted(visible):
        cursor += len(json.dumps(key, ensure_ascii=False)) + 2  # colon + space
        fields[key] = origin + cursor + 1  # opening string quote
        cursor += len(json.dumps(visible[key], ensure_ascii=False)) + 2

    def positions(field, a, b):
        raw = visible[field]
        # JSON escapes may expand a character; token indices are not character indices.
        start = fields[field] + len(json.dumps(raw[:a], ensure_ascii=False)) - 2
        end = fields[field] + len(json.dumps(raw[:b], ensure_ascii=False)) - 2
        result = [i for i, (x, y) in enumerate(offsets) if start <= x < y <= end]
        if not result:
            raise ValueError(f"Empty token span in {field}; no silent dropping")
        return result

    spans = sentence_spans(visible["facts"])
    meta = {"prompt_length": len(ids),
            "sentences": [positions("facts", a, b) for a, b in spans],
            "charge": positions("charge", 0, len(visible["charge"])),
            "target": positions("target_person", 0, len(visible["target_person"])) if "target_person" in visible else []}
    return ids, meta, spans


def training_rows(tokenizer, protocol, direct, bound, max_length):
    if [r["id"] for r in direct] != [r["id"] for r in bound] or any(r["split"] != "train" for r in direct + bound):
        raise ValueError("Auxiliary labels must be paired training cases")
    prepared, counts, by_charge = [], Counter(), defaultdict(Counter)
    for row, annotation in zip(direct, bound):
        answer = json.loads(row["completion"][0]["content"])
        teacher = json.loads(annotation["completion"][0]["content"])
        if answer != {k: teacher[k] for k in ("reasoning", "sentence_months")}:
            raise ValueError("Direct/bound supervision mismatch")
        visible = json.loads(row["prompt"][-1]["content"])
        if visible != json.loads(annotation["prompt"][-1]["content"]):
            raise ValueError("Direct/bound input mismatch")
        ids, meta, spans = prompt_features(tokenizer, row["prompt"])
        tokens = tokenize_sft_row(tokenizer, row, protocol)
        if tokens["input_ids"][:len(ids)] != ids or len(tokens["input_ids"]) > max_length:
            raise ValueError(f"Prefix mismatch/overlength: {row['id']}; no truncation")
        labels, diagnostics = weak_roles(visible["facts"], spans, teacher["evidence"])
        counts.update(diagnostics)
        by_charge[visible["charge"]].update(diagnostics)
        prepared.append({"id": row["id"], "input_ids": tokens["input_ids"], "labels": tokens["labels"],
                         "meta": meta, "role_labels": labels})
    report = {"version": VERSION, "cases": len(prepared), "roles": ROLES, "alignment": dict(counts),
              "by_charge": {k: dict(v) for k, v in by_charge.items()},
              "total_tokens_per_epoch": sum(len(r["input_ids"]) for r in prepared),
              "supervised_tokens_per_epoch": sum(sum(t != -100 for t in r["labels"]) for r in prepared),
              "max_tokens": max(len(r["input_ids"]) for r in prepared),
              "semantic_accuracy": None, "note": "Unaligned/conflicting spans ignored only in auxiliary loss; all SFT cases retained"}
    return prepared, report
