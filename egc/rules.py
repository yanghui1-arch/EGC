from __future__ import annotations

from .io import digest, index_unique
from .data import visible_case

STATES = {"supported", "refuted", "unknown"}


def conjunction(states):
    return "refuted" if "refuted" in states else "unknown" if "unknown" in states else "supported"


def disjunction(states):
    return "supported" if "supported" in states else "unknown" if "unknown" in states else "refuted"


def negate(state):
    return {"supported": "refuted", "refuted": "supported", "unknown": "unknown"}[state]


def expression_ids(expr):
    if isinstance(expr, str):
        return {expr}
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError("Expression must be a condition ID or a single all/any/not operator")
    op, value = next(iter(expr.items()))
    if op == "not":
        return expression_ids(value)
    if op not in {"all", "any"} or not isinstance(value, list) or not value:
        raise ValueError("all/any must have a nonempty expression list")
    return set().union(*(expression_ids(v) for v in value))


def evaluate(expr, states):
    if isinstance(expr, str):
        return states.get(expr, "unknown")
    op, value = next(iter(expr.items()))
    if op == "not":
        return negate(evaluate(value, states))
    children = [evaluate(v, states) for v in value]
    return conjunction(children) if op == "all" else disjunction(children)


def validate_rules(bundle, allow_draft=False):
    if bundle.get("schema_version") != 1:
        raise ValueError("Expected rules schema_version=1")
    conditions = index_unique(bundle["conditions"])
    rules = index_unique(bundle["rules"])
    if not rules or not conditions:
        raise ValueError("Empty rules or conditions")
    for condition in conditions.values():
        if not isinstance(condition.get("text"), str) or not condition["text"].strip():
            raise ValueError("Condition text is required")
    for rule in rules.values():
        if not expression_ids(rule["when"]) <= conditions.keys():
            raise ValueError(f"Unknown condition in rule {rule['id']}")
        for field in ("requires", "blocked_by"):
            refs = rule.get(field, [])
            if not isinstance(refs, list) or not set(refs) <= rules.keys():
                raise ValueError(f"Invalid {field} in {rule['id']}")
        if not rule.get("charges") or not isinstance(rule["charges"], list):
            raise ValueError("Every rule requires explicit charges or ['*']")
        if not rule.get("source") or not rule.get("version") or not rule.get("effect"):
            raise ValueError("Every rule requires source, version, effect")
        if not allow_draft and (rule.get("review_status") != "approved" or not rule.get("reviewer")):
            raise ValueError(f"Rule {rule['id']} needs named human review; --allow-draft is exploratory only")
    visiting, visited = set(), set()
    def visit(identifier):
        if identifier in visiting:
            raise ValueError("Cyclic rule dependencies/defeaters are unsupported in the pilot")
        if identifier in visited:
            return
        visiting.add(identifier)
        for ref in rules[identifier].get("requires", []) + rules[identifier].get("blocked_by", []):
            visit(ref)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in rules:
        visit(identifier)
    return bundle


def select_rules(bundle, charge):
    rules = {r["id"]: r for r in bundle["rules"]}
    chosen = {identifier for identifier, r in rules.items() if charge in r["charges"] or "*" in r["charges"]}
    pending = list(chosen)
    while pending:
        rule = rules[pending.pop()]
        for ref in rule.get("requires", []) + rule.get("blocked_by", []):
            if ref not in chosen:
                chosen.add(ref)
                pending.append(ref)
    selected = [r for r in bundle["rules"] if r["id"] in chosen]
    if not selected:
        raise ValueError(f"No rules cover given charge: {charge}")
    used = set().union(*(expression_ids(r["when"]) for r in selected))
    return {"schema_version": 1, "conditions": [c for c in bundle["conditions"] if c["id"] in used], "rules": selected}


def annotation_fingerprint(row, bundle):
    return digest({"input": visible_case(row), "rules": bundle})


def validate_annotation(annotation, row, bundle):
    """Validate offsets, coverage and provenance. This does NOT establish semantic truth."""
    if annotation.get("id") != row["id"]:
        raise ValueError("Annotation case ID mismatch")
    if annotation.get("fingerprint") != annotation_fingerprint(row, bundle):
        raise ValueError("Annotation input/rule fingerprint mismatch; re-annotate changed inputs")
    conditions = {c["id"] for c in bundle["conditions"]}
    observed = index_unique(annotation.get("conditions", []), "condition_id")
    if observed.keys() != conditions:
        raise ValueError("Annotation must cover each selected condition exactly once")
    facts = row["facts"]
    for item in observed.values():
        state = item.get("status")
        if state not in STATES:
            raise ValueError("Invalid condition state")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list) or (state != "unknown" and not evidence):
            raise ValueError("Supported/refuted conditions require source evidence")
        for span in evidence:
            start, end = span.get("start"), span.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(facts):
                raise ValueError("Invalid evidence offsets")
            if facts[start:end] != span.get("quote"):
                raise ValueError("Evidence must exactly match the original input substring")
    return annotation


def resolve_quotes(payload, row, bundle):
    """The API returns quotes; local code computes Python Unicode character offsets."""
    conditions = []
    for item in payload["conditions"]:
        spans = []
        for quote in item.get("quotes", []):
            if not isinstance(quote, str) or not quote:
                raise ValueError("Empty/invalid evidence quote")
            first = row["facts"].find(quote)
            if first < 0:
                raise ValueError("API produced a quote absent from the input")
            if row["facts"].find(quote, first + 1) >= 0:
                raise ValueError("Ambiguous quote; return a longer unique source span")
            spans.append({"start": first, "end": first + len(quote), "quote": quote})
        conditions.append({"condition_id": item["condition_id"], "status": item["status"], "evidence": spans})
    annotation = {"id": row["id"], "fingerprint": annotation_fingerprint(row, bundle),
                  "review_status": "draft", "conditions": conditions}
    return validate_annotation(annotation, row, bundle)


def compose(bundle, annotation):
    """Three-valued dependency composition; effects are descriptors, never numeric clipping."""
    rules = index_unique(bundle["rules"])
    states = {c["condition_id"]: c["status"] for c in annotation["conditions"]}
    cache = {}
    def state(identifier):
        if identifier not in cache:
            rule = rules[identifier]
            parts = [evaluate(rule["when"], states)]
            parts += [state(ref) for ref in rule.get("requires", [])]
            parts += [negate(state(ref)) for ref in rule.get("blocked_by", [])]
            cache[identifier] = conjunction(parts)
        return cache[identifier]
    trace = [{"rule_id": identifier, "status": state(identifier), "effect": rule["effect"],
              "requires": rule.get("requires", []), "blocked_by": rule.get("blocked_by", [])}
             for identifier, rule in rules.items()]
    return {"rules": trace,
            "unknown_conditions": sorted(c for c, status in states.items() if status == "unknown"),
            "warning": "Statuses are conditional on annotation and rule correctness, not a proof of legal truth."}


def import_chains(text, charge, source, version):
    """Preserve compound situation text; do not pretend AND/OR have been legally parsed."""
    conditions, rules = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(" -> ")]
        if len(parts) != 3:
            raise ValueError("Expected premise -> situation -> conclusion")
        rid = "chain_" + digest([charge, parts])[:16]
        conditions += [{"id": rid + "_p", "text": parts[0]}, {"id": rid + "_s", "text": parts[1]}]
        rules.append({"id": rid, "charges": [charge], "when": {"all": [rid + "_p", rid + "_s"]},
                      "requires": [], "blocked_by": [], "effect": {"kind": "statutory_outcome", "text": parts[2]},
                      "source": source, "version": version, "review_status": "draft", "reviewer": None})
    return validate_rules({"schema_version": 1, "conditions": conditions, "rules": rules}, allow_draft=True)
