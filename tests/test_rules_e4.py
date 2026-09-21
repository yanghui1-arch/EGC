"""Synthetic mechanism tests, not empirical legal validation."""
import json
from pathlib import Path
import unittest

from egc.prompts import make_prompt
from egc.rules import annotation_fingerprint, compose, select_rules, validate_rules

BUNDLE = json.loads((Path(__file__).parents[1] / "configs/rules_e4_draft.json").read_text(encoding="utf-8"))


def resolve(charge, **states):
    selected = select_rules(BUNDLE, charge)
    annotation = {"conditions": [{"condition_id": c["id"], "status": states.get(c["id"], "unknown")}
                                 for c in selected["conditions"]]}
    return {r["rule_id"]: r["status"] for r in compose(selected, annotation)["rules"]}


class E4RuleMechanismTests(unittest.TestCase):
    def test_draft_only_and_unknown_not_success(self):
        validate_rules(BUNDLE, allow_draft=True)
        with self.assertRaises(ValueError): validate_rules(BUNDLE)
        for charge in ("诈骗罪", "抢劫罪"):
            self.assertEqual(set(resolve(charge).values()), {"unknown"})

    def test_surrender_exception_restoration_and_downstream(self):
        states = dict(scope_surrender="supported", legal_voluntary_surrender="supported",
                      main_facts_disclosed="supported", joint_disclosure_satisfied="supported",
                      identity_disclosure_satisfied="supported", escaped_after_surrender="refuted",
                      retracted_after_disclosure="supported", truth_restored_before_judgment="refuted",
                      crime_minor_verified="supported")
        self.assertEqual(resolve("诈骗罪", **states)["surrender_minor_exemption"], "refuted")
        states["truth_restored_before_judgment"] = "unknown"
        self.assertEqual(resolve("诈骗罪", **states)["surrender_leniency"], "unknown")
        states["truth_restored_before_judgment"] = "supported"
        self.assertEqual(resolve("诈骗罪", **states)["surrender_minor_exemption"], "supported")
        states["escaped_after_surrender"] = "supported"
        self.assertEqual(resolve("诈骗罪", **states)["ordinary_surrender"], "refuted")

    def test_e3_labels_do_not_satisfy_stronger_conditions(self):
        states = dict(voluntary_appearance="supported", truthful_confession="supported",
                      returned_or_compensated="supported", victim_forgiveness="supported")
        result = resolve("诈骗罪", **states)
        self.assertEqual(result["ordinary_surrender"], "unknown")
        self.assertEqual(result["fraud_article3_two_branches"], "unknown")

    def test_fraud_prerequisites_and_alternative_branches(self):
        states = dict(scope_fraud_article3="supported", admitted_guilt="supported", repented="unknown",
                      full_restitution_before_judgment="supported", target_victim_forgiveness="refuted")
        self.assertEqual(resolve("诈骗罪", **states)["fraud_article3_two_branches"], "unknown")
        states.update(repented="supported", full_restitution_before_judgment="unknown")
        self.assertEqual(resolve("诈骗罪", **states)["fraud_article3_two_branches"], "unknown")
        states["target_victim_forgiveness"] = "supported"
        self.assertEqual(resolve("诈骗罪", **states)["fraud_article3_two_branches"], "supported")
        states["scope_fraud_article3"] = "unknown"
        self.assertEqual(resolve("诈骗罪", **states)["fraud_article3_two_branches"], "unknown")

    def test_small_taxi_exception_and_home_missing_intent(self):
        states = dict(scope_robbery_2016="supported", qualifying_passenger_transport="supported",
                      transport_operating="supported", robbery_on_or_intercepting_transport="supported",
                      transport_victim_in_scope="supported", small_taxi="supported")
        self.assertEqual(resolve("抢劫罪", **states)["public_transport_robbery_qualifier"], "refuted")
        states["small_taxi"] = "unknown"
        self.assertEqual(resolve("抢劫罪", **states)["public_transport_robbery_qualifier"], "unknown")
        states["small_taxi"] = "refuted"
        self.assertEqual(resolve("抢劫罪", **states)["public_transport_robbery_qualifier"], "supported")
        states.update(home_premises_verified="supported", robbery_inside_home="supported",
                      lawful_entry_then_new_intent="refuted")
        self.assertEqual(resolve("抢劫罪", **states)["home_robbery_qualifier"], "unknown")

    def test_concat_egc_same_information_except_computed_trace(self):
        row = {"id": "synthetic", "facts": "纯虚构的机制测试，无案件真值。", "charge": "诈骗罪"}
        selected = select_rules(BUNDLE, row["charge"])
        annotation = {"id": row["id"], "fingerprint": annotation_fingerprint(row, selected),
                      "conditions": [{"condition_id": c["id"], "status": "unknown", "evidence": []}
                                     for c in selected["conditions"]]}
        flat = make_prompt(row, "concat", BUNDLE, annotation, allow_draft=True)
        egc = make_prompt(row, "egc", BUNDLE, annotation, allow_draft=True)
        a, b = json.loads(flat[1]["content"]), json.loads(egc[1]["content"])
        self.assertIn("composed_rule_trace", b)
        del b["composed_rule_trace"]
        self.assertEqual(a, b)
        self.assertEqual(flat[0], egc[0])


if __name__ == "__main__": unittest.main()
