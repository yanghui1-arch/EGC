import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from egc import facts, facts_v2
from egc.annotate import ApiRequestError, call_deepseek
from egc.fact_experiment import run, report_run
from egc.io import read_json, read_rows
from test_fact_experiment import case, envelope

PROFILE = {"conditions": [{"id": c, "text": c} for c in dict.fromkeys(v[0] for v in facts_v2.EVENTS.values())]}


def event(kind, actor="defendant", scope="current_case", coverage="case_wide", ids=None):
    return {"kind": kind, "actor": actor, "scope": scope, "coverage": coverage, "unit_ids": ids or ["s1"]}


def bound(events, source=None):
    return facts.validate({"observations": events}, source or case(), PROFILE, "bound_v2")


def states(result):
    return {c["condition_id"]: c["status"] for c in result["annotation"]["conditions"]}


class RevisedFactTests(unittest.TestCase):
    def test_units_lossless_and_repeated_names_have_distinct_offsets(self):
        text = "甲归案。甲归案。\n甲的家属赔偿；余额未偿还"
        units = facts_v2.source_units(text)
        self.assertEqual(''.join(u['text'] for u in units), text)
        for u in units:
            self.assertEqual(text[u['start']:u['end']], u['text'])
        result = bound([event("completed_repayment", "proxy", ids=["s3"])], case(text=text))
        ev = result['annotation']['conditions'][2]['evidence'][0]
        self.assertEqual(text[ev['start']:ev['end']], ev['quote'])

    def test_recovery_is_nondecisive_and_cannot_refute_arrival(self):
        got = states(bound([event("property_recovery")]))
        self.assertEqual(got['returned_or_compensated'], 'unknown')
        self.assertEqual(got['voluntary_appearance'], 'unknown')

    def test_arrival_and_summons_do_not_imply_arrest(self):
        for kind in ('arrival_unspecified', 'summoned_arrival'):
            self.assertEqual(states(bound([event(kind)]))['voluntary_appearance'], 'unknown')
        self.assertEqual(states(bound([event('arrest')]))['voluntary_appearance'], 'refuted')

    def test_empty_observations_are_unknown_and_never_negative(self):
        self.assertEqual(set(states(bound([])).values()), {'unknown'})
        self.assertEqual({d['reason'] for d in bound([])['derivation']}, {'not_mentioned'})

    def test_proxy_other_actor_wrong_scope_and_unclear_binding(self):
        self.assertEqual(states(bound([event('completed_repayment','proxy')]))['returned_or_compensated'], 'supported')
        for obs in [event('completed_repayment','other'), event('completed_repayment',scope='prior_case'),
                    event('completed_repayment','unclear')]:
            self.assertEqual(states(bound([obs]))['returned_or_compensated'], 'unknown')

    def test_conflict_order_invariant_and_nondecisive_does_not_override(self):
        obs = [event('completed_repayment'), event('explicit_nonpayment')]
        self.assertEqual(states(bound(obs)), states(bound(list(reversed(obs)))))
        self.assertEqual(states(bound(obs))['returned_or_compensated'], 'unknown')
        self.assertEqual(states(bound([event('completed_repayment'),event('property_recovery')]))['returned_or_compensated'], 'supported')

    def test_prior_case_and_full_case_gates(self):
        self.assertEqual(states(bound([event('prior_conviction')]))['prior_conviction'], 'unknown')
        self.assertEqual(states(bound([event('prior_conviction',scope='prior_case')]))['prior_conviction'], 'supported')
        self.assertEqual(states(bound([event('outcome_failed',coverage='single_event')]))['unsuccessful_completion'], 'unknown')
        self.assertEqual(states(bound([event('outcome_failed'),event('outcome_achieved',coverage='single_event')]))['unsuccessful_completion'], 'unknown')

    def test_missing_duplicate_unknown_and_nonstring_references_fail(self):
        for ids in ([], ['s99'], ['s1','s1'], [1], 's1'):
            obs = event('arrest'); obs['unit_ids'] = ids
            with self.assertRaises(ValueError):
                bound([obs])

    def test_unknown_event_and_bad_binding_and_budget_fail(self):
        for observations in ([event('invented')], [event('arrest','invented')], [event('arrest')]*4):
            with self.assertRaises(ValueError):
                bound(observations)

    def test_flat_coverage_and_unknown_evidence(self):
        body = {'conditions':[{'condition_id': c['id'], 'status':'unknown', 'unit_ids':[]} for c in PROFILE['conditions']]}
        self.assertEqual(set(states(facts.validate(body,case(),PROFILE,'flat_v2')).values()), {'unknown'})
        body['conditions'][0]['unit_ids'] = ['s1']
        with self.assertRaises(ValueError):
            facts.validate(body,case(),PROFILE,'flat_v2')
        body['conditions'].pop()
        with self.assertRaises(ValueError):
            facts.validate(body,case(),PROFILE,'flat_v2')

    def test_labels_ids_expectations_not_in_payload_and_shared_context(self):
        source = case(); changed = {**source, 'id':'changed', 'sentence_months':999, 'opinion':'hidden', 'expected':'hidden'}
        for mode in facts_v2.MODES:
            self.assertEqual(facts.payload(source,PROFILE,mode,'test',100),facts.payload(changed,PROFILE,mode,'test',100))
        a,b=[facts.payload(source,PROFILE,m,'test',100) for m in facts_v2.MODES]
        self.assertEqual(a['messages'][1],b['messages'][1])
        self.assertTrue(all(p['messages'][0]['content'].startswith(facts_v2.BOUNDARIES) for p in (a,b)))

    def test_incorrect_event_semantics_are_not_claimed_to_be_verified(self):
        # Deliberately incorrect teacher type still passes structural gates. This limit is explicit.
        result = bound([event('completed_repayment')],case(text='仅有财物被查扣。'))
        self.assertEqual(states(result)['returned_or_compensated'],'supported')
        self.assertEqual(result['quality'],'structural_only_not_gold')


class RevisedRunTests(unittest.TestCase):
    def test_mocked_two_mode_run_cache_report_full_review_and_version_isolation(self):
        def respond(request,key,metadata,transport,capture_response):
            flat = '逐条件直接判断' in request['messages'][0]['content']
            content = ({'conditions':[{'condition_id': c['id'], 'status':'unknown', 'unit_ids':[]} for c in PROFILE['conditions']]}
                       if flat else {'observations':[]})
            capture_response(envelope(content))
            return content, {}, {}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'DEEPSEEK_API_KEY':'fixture'}), \
                patch('egc.fact_experiment.call_deepseek',side_effect=respond) as call:
            result=run([case()],PROFILE,tmp,modes=facts_v2.MODES,limit=2)
            self.assertEqual(result['statuses'],{'accepted':2})
            self.assertEqual(result['version'],facts_v2.VERSION)
            run([case()],PROFILE,tmp,modes=facts_v2.MODES,limit=2)
            self.assertEqual(call.call_count,2)
            report=report_run([case()],tmp,Path(tmp)/'comparison.json')
            self.assertEqual(report['paired']['flat_v2_vs_bound_v2']['valid_pairs'],1)
            self.assertEqual(report['review_items'],8)
            self.assertTrue(all(r['review_state'] is None for r in read_rows(Path(tmp)/'review_queue.jsonl')))
            with self.assertRaisesRegex(ValueError,'identity'):
                run([case()],PROFILE,tmp,modes=['flat'],dry_run=True)

    def test_request_failure_diagnostics_are_sanitized_and_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'DEEPSEEK_API_KEY':'fixture'}), \
                patch('egc.fact_experiment.call_deepseek',side_effect=ApiRequestError('http_status',401)) as call:
            run([case()],PROFILE,tmp,modes=['flat_v2'])
            run([case()],PROFILE,tmp,modes=['flat_v2'])
            self.assertEqual(call.call_count,1)
            self.assertEqual(read_rows(Path(tmp)/'records.jsonl')[0]['diagnostic'],{'category':'http_status','code':401})
            report=report_run([case()],tmp,Path(tmp)/'comparison.json')
            self.assertEqual(report['review_items'],8)
            self.assertTrue(all(r['predictions']['flat_v2']['state'] is None for r in read_rows(Path(tmp)/'review_queue.jsonl')))

    def test_curl_failure_omits_body_stderr_key_and_raw_status(self):
        for completed,category,code in [
            (SimpleNamespace(returncode=6,stdout='',stderr='secret header'), 'curl_exit',6),
            (SimpleNamespace(returncode=0,stdout='secret body\n401'),'http_status',401),
            (SimpleNamespace(returncode=0,stdout='secret body\nsecret status'),'http_status',None),
            (SimpleNamespace(returncode=0,stdout='secret body\n200'),'invalid_http_envelope',None)]:
            with patch('subprocess.run',return_value=completed), self.assertRaises(ApiRequestError) as ctx:
                call_deepseek({},'secret key',transport='curl')
            self.assertNotIn('secret',str(ctx.exception))
            self.assertEqual(ctx.exception.diagnostic.get('code'),code)
            self.assertEqual(ctx.exception.diagnostic['category'],category)

    def test_cli_nonzero_when_terminal_status_contains_failures(self):
        from egc.cli import main
        with patch('sys.argv',['egc','facts-run','--input','unused','--profile','unused','--output-dir','unused']), \
                patch('egc.cli.run',return_value={'statuses':{'api_error':1},'complete':True}), self.assertRaises(SystemExit) as ctx:
            main()
        self.assertEqual(ctx.exception.code,3)


if __name__ == '__main__':
    unittest.main()
