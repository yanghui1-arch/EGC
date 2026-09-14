import json
import math
import random
import re

from .io import digest, index_unique


def parse_output(text):
    """Strict structured output: no guessed numeric fallback or failure-as-zero."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("reasoning"), str) or not value["reasoning"].strip():
        return None
    months = value.get("sentence_months")
    if type(months) is not int or months < 0:
        return None
    return value


def char_rouge_l(reference, hypothesis):
    # Explicitly character-level diagnostic, NOT the paper's tokenized ROUGE-L.
    a, b = re.sub(r"\s+", "", reference), re.sub(r"\s+", "", hypothesis)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j-1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def errors(rows, predictions):
    gold, predicted = index_unique(rows), index_unique(predictions)
    if not set(predicted) <= set(gold):
        raise ValueError("Predictions contain unknown case IDs")
    result = []
    for identifier, row in gold.items():
        pred = predicted.get(identifier)
        parsed = parse_output(pred.get("text", "")) if pred else None
        if pred and pred.get("finish_reason") == "length":
            parsed = None
        result.append({"id": identifier, "charge": row["charge"], "valid": parsed is not None,
                       "target": row["sentence_months"],
                       "prediction": parsed["sentence_months"] if parsed else None,
                       "absolute_error": abs(parsed["sentence_months"] - row["sentence_months"]) if parsed else None,
                       "char_rouge_l_reasoning": char_rouge_l(row["opinion"], parsed["reasoning"]) if parsed and row.get("opinion") else None})
    return result


def summarize(rows, predictions):
    detail = errors(rows, predictions)
    valid = [d for d in detail if d["valid"]]
    ae = [d["absolute_error"] for d in valid]
    rouge = [d["char_rouge_l_reasoning"] for d in valid if d["char_rouge_l_reasoning"] is not None]
    mae = sum(ae) / len(ae) if ae else None
    rmse = math.sqrt(sum(e*e for e in ae) / len(ae)) if ae else None
    return {"dataset_hash": digest(rows), "n": len(rows), "valid": len(valid),
            "invalid_or_missing": len(rows) - len(valid), "coverage": len(valid)/len(rows),
            "mae_months_full": mae if len(valid) == len(rows) else None,
            "rmse_months_full": rmse if len(valid) == len(rows) else None,
            "mae_months_valid_only": mae, "rmse_months_valid_only": rmse,
            "char_rouge_l_reasoning_valid_only": sum(rouge)/len(rouge) if rouge else None,
            "char_rouge_n": len(rouge), "eligible_for_full_mae_comparison": len(valid) == len(rows),
            "per_case": detail,
            "limitations": "Character ROUGE is diagnostic only; no automatic legal correctness or official LawBench score."}


def compare(rows, baseline, candidate, seed=42, samples=2000):
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    first, second = errors(rows, baseline), errors(rows, candidate)
    if any(not x["valid"] for x in first + second):
        raise ValueError("Full coverage required for paired MAE comparison; report failures separately")
    differences = [b["absolute_error"] - a["absolute_error"] for a, b in zip(first, second)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(samples))
    return {"n": len(rows), "delta_mae_candidate_minus_baseline": sum(differences)/len(differences),
            "paired_case_bootstrap_ci95": [means[int(samples*.025)], means[min(samples-1, int(samples*.975))]],
            "seed": seed, "bootstrap_samples": samples,
            "note": "Assumes independent cases; remove/group same-case records first. Does not capture training-seed variance."}
