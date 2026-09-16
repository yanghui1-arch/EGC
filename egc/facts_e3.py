"""Frozen seven-condition pilot with uncertainty context kept outside evidence."""
import copy
import json
from collections import Counter

from . import facts_v2
from .io import digest, index_unique

VERSION = "fact-seven-context-e3-v1"
MODES = ("flat_e3", "bound_e3")
CONDITION_IDS = frozenset(v[0] for v in facts_v2.EVENTS.values()) - {"unsuccessful_completion"}


def check_profile(profile):
    if set(index_unique(profile["conditions"])) != CONDITION_IDS:
        raise ValueError("E3 requires exactly the frozen seven conditions, excluding unsuccessful_completion")


def check_protocol(rows, profile, modes, model, max_tokens, protocol):
    check_profile(profile)
    if not isinstance(protocol, dict) or protocol.get("version") != VERSION:
        raise ValueError("E3 requires a frozen --protocol file")
    actual = {"source_hash": digest(rows), "profile_hash": digest(profile), "cases": len(rows),
              "charges": dict(Counter(r["charge"] for r in rows)), "modes": list(modes),
              "model": model, "max_tokens": max_tokens, "total_requests": len(rows) * len(modes)}
    if list(modes) != list(MODES) or any(protocol.get(k) != v for k, v in actual.items()):
        raise ValueError("E3 inputs/settings differ from frozen protocol; do not substitute samples")
    if any(r["split"] != "train" for r in rows):
        raise ValueError("E3 accepts frozen train cases only")


def payload(row, profile, mode, model, max_tokens):
    check_profile(profile)
    legacy_mode = {"flat_e3": "flat_v2", "bound_e3": "bound_v2"}.get(mode)
    if legacy_mode is None:
        raise ValueError("Unknown E3 mode")
    result = facts_v2.payload(row, profile, legacy_mode, model, max_tokens)
    user = json.loads(result["messages"][1]["content"])
    user["event_definitions"] = {k: v for k, v in user["event_definitions"].items()
                                 if v["condition_id"] in CONDITION_IDS}
    result["messages"][1]["content"] = json.dumps(user, ensure_ascii=False)
    prompt = result["messages"][0]["content"]
    if mode == "flat_e3":
        old = "unknown引用为空；其它状态必须引用支持该判断的句段。"
        if old not in prompt:
            raise ValueError("Frozen flat prompt changed; review E3 protocol before running")
        prompt = prompt.replace(old, "unknown可引用解释不确定性的上下文，也可为空；这些引用不是支持或反证。"
                                "其它状态必须引用支持该判断的句段。")
    result["messages"][0]["content"] = prompt + "本轮仅处理列入conditions和event_definitions的七项条件/相关事件。"
    return result


def validate(body, row, profile, mode):
    check_profile(profile)
    context = {cid: [] for cid in CONDITION_IDS}
    if mode == "flat_e3":
        # Preserve the original body/cache. Only the downstream evidence projection is empty.
        projected = copy.deepcopy(body)
        units = index_unique(facts_v2.source_units(row["facts"]))
        for decision in projected["conditions"]:
            cid = decision["condition_id"]
            if cid not in CONDITION_IDS:
                raise ValueError("Condition outside E3 scope")
            if decision["status"] == "unknown":
                context[cid] = facts_v2.evidence(decision["unit_ids"], units, required=False)
                decision["unit_ids"] = []
        result = facts_v2.validate(projected, row, profile, "flat_v2")
    elif mode == "bound_e3":
        result = facts_v2.validate(body, row, profile, "bound_v2")
        states = {c["condition_id"]: c["status"] for c in result["annotation"]["conditions"]}
        for obs in result["observations"]:
            cid = obs["condition_id"]
            if states[cid] == "unknown":
                for span in obs["evidence"]:
                    if span not in context[cid]:
                        context[cid].append(span)
        for trace in result["derivation"]:
            if trace["reason"] == "not_mentioned":
                trace["reason"] = "no_observation_extracted"
    else:
        raise ValueError("Unknown E3 mode")
    result["uncertainty_context"] = context
    result["protocol_version"] = VERSION
    result["limitation"] += " Uncertainty context is not supporting evidence; missing observations do not prove missing facts."
    return result


def add_report_details(report, rows, records, modes, profile):
    """Report all denominators by condition and charge; no synthetic accuracy score."""
    charges = {r["id"]: r["charge"] for r in rows}
    per_condition, per_charge = {}, {}
    for mode in modes:
        group = [r for r in records if r["mode"] == mode]
        per_condition[mode] = {}
        for condition in profile["conditions"]:
            cid = condition["id"]
            states, contexts = Counter(), 0
            for record in group:
                if record["status"] != "accepted":
                    states[record["status"]] += 1
                    continue
                state = next(c["status"] for c in record["result"]["annotation"]["conditions"] if c["condition_id"] == cid)
                states[state] += 1
                contexts += bool(record["result"].get("uncertainty_context", {}).get(cid))
            per_condition[mode][cid] = {"requested_cases": len(rows), "states_or_failures": dict(states),
                                       "unknown_with_context": contexts}
        per_charge[mode] = {}
        for charge in sorted(set(charges.values())):
            subset = [r for r in group if charges[r["id"]] == charge]
            per_charge[mode][charge] = {"requested_cases": len(subset),
                                       "record_statuses": dict(Counter(r["status"] for r in subset))}
    report.update(per_condition=per_condition, per_charge=per_charge,
                  excluded_condition="unsuccessful_completion", hypotheses_confirmed=False)
