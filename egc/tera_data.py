"""TERA prompt-only spans plus sparse train-only role and target-mention supervision (no API)."""
import json
import re
from collections import Counter, defaultdict

from .server import render_chat, tokenize_sft_row

VERSION = "tera-latent-target-v2"
LEGACY_VERSION = "tera-sentence-memory-v1"  # explicit target_person input; inference of saved runs only
ROLES = ("target", "other", "uncertain")
PLACEHOLDERS = "某X×*"


def target_mode(version):
    if version not in (VERSION, LEGACY_VERSION):
        raise ValueError("Unknown TERA module version")
    return "inferred" if version == VERSION else "explicit"


def target_char_spans(facts, name):
    """Literal name occurrences; label-side only. Skips 张某 inside another person's 张某某/张某X."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing target name")
    spans, start = [], facts.find(name)
    while start >= 0:
        end = start + len(name)
        if not (end < len(facts) and facts[end] in PLACEHOLDERS):
            spans.append([start, end])
        start = facts.find(name, end)
    return spans


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


def prompt_features(tokenizer, messages, target_spans=None):
    """Offsets come from the serialized user JSON, never from an answer search.

    target_spans are label-only character spans in facts (train weak supervision or dev diagnostics)."""
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
    if target_spans is not None:
        if any(not 0 <= a < b <= len(visible["facts"]) for a, b in target_spans):
            raise ValueError("Target mention outside facts")
        meta["target_mentions"] = sorted({i for a, b in target_spans for i in positions("facts", a, b)})
    return ids, meta, spans


def training_rows(tokenizer, protocol, direct, bound, max_length, balanced_sentence=False, targets=None):
    if [r["id"] for r in direct] != [r["id"] for r in bound] or any(r["split"] != "train" for r in direct + bound):
        raise ValueError("Auxiliary labels must be paired training cases")
    if targets is not None and set(targets) != {r["id"] for r in direct}:
        raise ValueError("Target-mention supervision must cover exactly the training cases")
    prepared, counts, by_charge = [], Counter(), defaultdict(Counter)
    for row, annotation in zip(direct, bound):
        answer = json.loads(row["completion"][0]["content"])
        teacher = json.loads(annotation["completion"][0]["content"])
        if answer != {k: teacher[k] for k in ("reasoning", "sentence_months")}:
            raise ValueError("Direct/bound supervision mismatch")
        visible = json.loads(row["prompt"][-1]["content"])
        if visible != json.loads(annotation["prompt"][-1]["content"]):
            raise ValueError("Direct/bound input mismatch")
        ids, meta, spans = prompt_features(tokenizer, row["prompt"], None if targets is None else targets[row["id"]])
        if targets is not None:
            counts["target_mention_cases" if meta["target_mentions"] else "no_target_mention_cases"] += 1
        tokens = tokenize_sft_row(tokenizer, row, protocol, include_offsets=balanced_sentence)
        if tokens["input_ids"][:len(ids)] != ids or len(tokens["input_ids"]) > max_length:
            raise ValueError(f"Prefix mismatch/overlength: {row['id']}; no truncation")
        labels, diagnostics = weak_roles(visible["facts"], spans, teacher["evidence"])
        counts.update(diagnostics)
        by_charge[visible["charge"]].update(diagnostics)
        item = {"id": row["id"], "input_ids": tokens["input_ids"], "labels": tokens["labels"],
                "meta": meta, "role_labels": labels}
        if balanced_sentence:
            body = row["completion"][0]["content"]
            months = answer["sentence_months"]
            tail = ', "sentence_months": ' + str(months) + '}'
            if (type(months) is not int or months < 0 or set(answer) != {"reasoning", "sentence_months"}
                    or body != json.dumps(answer, ensure_ascii=False) or not body.endswith(tail)):
                raise ValueError("Expected canonical reasoning/month answer for numerical supervision")
            start = tokens["completion_char_start"] + len(body) - 1 - len(str(months))
            end = start + len(str(months))
            mask = [x < end and y > start for x, y in tokens["offset_mapping"]]
            if (not any(mask) or any(mask[:len(ids)]) or all(mask[len(ids):])
                    or any(not any(x <= c < y for x, y in tokens["offset_mapping"]) for c in range(start, end))):
                raise ValueError("Invalid sentence-number token span; no silent fallback")
            item["sentence_mask"] = mask
        prepared.append(item)
    report = {"version": VERSION, "cases": len(prepared), "roles": ROLES, "alignment": dict(counts),
              "by_charge": {k: dict(v) for k, v in by_charge.items()},
              "total_tokens_per_epoch": sum(len(r["input_ids"]) for r in prepared),
              "supervised_tokens_per_epoch": sum(sum(t != -100 for t in r["labels"]) for r in prepared),
              "max_tokens": max(len(r["input_ids"]) for r in prepared),
              "semantic_accuracy": None, "note": "Unaligned/conflicting spans ignored only in auxiliary loss; all SFT cases retained"}
    return prepared, report
