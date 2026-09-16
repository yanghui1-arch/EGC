import copy
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egc import facts, facts_e3
from egc.fact_experiment import run, report_run
from egc.io import digest, read_json, read_rows, write_json, write_rows
from test_fact_experiment import case, envelope
from test_facts_v2 import event

PROFILE = read_json(Path(__file__).resolve().parents[1] / 'configs/evidence_profile_e3.json')


def protocol(rows):
    return {'version':facts_e3.VERSION, 'source_hash':digest(rows), 'profile_hash':digest(PROFILE),
            'cases':len(rows), 'charges':dict(Counter(r['charge'] for r in rows)),
            'modes':list(facts_e3.MODES), 'model':'test', 'max_tokens':256, 'total_requests':len(rows)*2}


def flat():
    return {'conditions':[{'condition_id':c['id'], 'status':'unknown', 'unit_ids':[]} for c in PROFILE['conditions']]}


class E3Tests(unittest.TestCase):
    def test_unknown_context_preserved_without_becoming_support_evidence_or_mutating_body(self):
        body=flat(); body['conditions'][0]['unit_ids']=['s1']; original=copy.deepcopy(body)
        result=facts.validate(body,case(),PROFILE,'flat_e3')
        self.assertEqual(body,original)
        c=result['annotation']['conditions'][0]
        self.assertEqual((c['status'],c['evidence']),('unknown',[]))
        self.assertEqual(result['uncertainty_context'][c['condition_id']][0]['quote'],case()['facts'])
        with self.assertRaisesRegex(ValueError,'Unknown must have no evidence'):
            facts.validate(body,case(),PROFILE,'flat_v2')

    def test_unknown_context_still_requires_valid_unique_unit_ids(self):
        for ids in (['not-real'], ['s1','s1'], [True], 's1'):
            body=flat(); body['conditions'][0]['unit_ids']=ids
            with self.assertRaises(ValueError):
                facts.validate(body,case(),PROFILE,'flat_e3')

    def test_nonunknown_requires_evidence_and_has_no_uncertainty_context(self):
        body=flat(); body['conditions'][2].update(status='supported',unit_ids=['s1'])
        result=facts.validate(body,case(),PROFILE,'flat_e3')
        self.assertTrue(result['annotation']['conditions'][2]['evidence'])
        self.assertEqual(result['uncertainty_context']['returned_or_compensated'],[])
        body['conditions'][2]['unit_ids']=[]
        with self.assertRaises(ValueError):
            facts.validate(body,case(),PROFILE,'flat_e3')

    def test_bound_context_and_no_observation_is_not_no_information(self):
        result=facts.validate({'observations':[event('property_recovery','unclear')]},case(),PROFILE,'bound_e3')
        c=next(c for c in result['annotation']['conditions'] if c['condition_id']=='returned_or_compensated')
        self.assertEqual((c['status'],c['evidence']),('unknown',[]))
        self.assertTrue(result['uncertainty_context']['returned_or_compensated'])
        self.assertEqual(result['derivation'][0]['reason'],'no_observation_extracted')

    def test_profile_and_event_scope_are_frozen_and_shared_and_labels_blind(self):
        a,b=[facts.payload(case(),PROFILE,m,'test',256) for m in facts_e3.MODES]
        self.assertEqual(a['messages'][1],b['messages'][1])
        context=json.loads(a['messages'][1]['content'])
        self.assertEqual(len(context['conditions']),7)
        self.assertNotIn('outcome_failed',context['event_definitions'])
        self.assertNotIn('outcome_achieved',context['event_definitions'])
        for m in facts_e3.MODES:
            changed={**case(),'sentence_months':999,'opinion':'hidden','expected':'hidden','id':'different'}
            self.assertEqual(facts.payload(case(),PROFILE,m,'test',256),facts.payload(changed,PROFILE,m,'test',256))
        with self.assertRaises(ValueError):
            facts.validate({'observations':[event('outcome_failed')]},case(),PROFILE,'bound_e3')
        with self.assertRaises(ValueError):
            facts.payload(case(),{'conditions':PROFILE['conditions'][:-1]},'flat_e3','test',256)

    def test_protocol_mismatch_refused_before_api_or_files(self):
        rows=[case()]; frozen=protocol(rows)
        with tempfile.TemporaryDirectory() as tmp, patch('egc.fact_experiment.call_deepseek') as api:
            out=Path(tmp)/'new'
            for key in ('source_hash','profile_hash','cases','charges','modes','model','max_tokens','total_requests'):
                changed={**frozen,key:None}
                with self.assertRaisesRegex(ValueError,'frozen protocol'):
                    run(rows,PROFILE,out,modes=facts_e3.MODES,model='test',max_tokens=256,protocol=changed)
            with self.assertRaisesRegex(ValueError,'requires a frozen'):
                run(rows,PROFILE,out,modes=facts_e3.MODES,model='test',max_tokens=256)
            self.assertFalse(out.exists()); api.assert_not_called()
            result=run(rows,PROFILE,out,modes=facts_e3.MODES,model='test',max_tokens=256,protocol=frozen,dry_run=True)
            self.assertTrue(result['protocol_verified']); self.assertEqual(result['conditions'],7)
            self.assertFalse(out.exists()); api.assert_not_called()

    def test_cli_mocked_e3_cache_and_full_review_failure_denominators(self):
        from egc.cli import parser, run as cli_run
        count=0
        def respond(request,key,metadata,transport,capture_response):
            nonlocal count
            count+=1
            if count==2:
                capture_response(envelope(finish='length'))
                raise ValueError('Incomplete')
            body=flat(); body['conditions'][0]['unit_ids']=['s1']
            capture_response(envelope(body))
            return body,{},{}
        rows=[case()]
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'DEEPSEEK_API_KEY':'fixture'}), \
                patch('egc.fact_experiment.call_deepseek',side_effect=respond) as api:
            root=Path(tmp); out=root/'run'
            write_rows(root/'input.jsonl',rows); write_json(root/'profile.json',PROFILE); write_json(root/'protocol.json',protocol(rows))
            args=parser().parse_args(['facts-run','--input',str(root/'input.jsonl'),'--profile',str(root/'profile.json'),
                '--protocol',str(root/'protocol.json'),'--output-dir',str(out),'--modes',*facts_e3.MODES,
                '--model','test','--max-tokens','256'])
            result=cli_run(args); self.assertEqual(result['statuses'],{'accepted':1,'rejected':1})
            cli_run(args); self.assertEqual(api.call_count,2)
            report=report_run(rows,out,out/'comparison.json')
            self.assertEqual(report['review_items'],7)
            self.assertEqual(report['paired']['flat_e3_vs_bound_e3']['valid_pairs'],0)
            for c in report['per_condition']['bound_e3'].values():
                self.assertEqual(c['states_or_failures'],{'rejected':1})
                self.assertEqual(c['requested_cases'],1)
            self.assertEqual(report['per_condition']['flat_e3']['voluntary_appearance']['unknown_with_context'],1)
            queue=read_rows(out/'review_queue.jsonl')
            self.assertTrue(queue[0]['predictions']['flat_e3']['uncertainty_context'])
            self.assertEqual(queue[0]['predictions']['flat_e3']['evidence'],[])
            self.assertIsNone(queue[0]['predictions']['bound_e3']['state'])
            altered=protocol(rows); altered['note']='changed protocol'
            with self.assertRaisesRegex(ValueError,'identity'):
                run(rows,PROFILE,out,modes=facts_e3.MODES,model='test',max_tokens=256,protocol=altered)

    def test_config_keeps_seven_definitions_verbatim_and_hash_matches_protocol(self):
        root=Path(__file__).resolve().parents[1]
        prior=read_json(root/'configs/evidence_profile_v3.json')
        self.assertEqual(PROFILE['conditions'],[c for c in prior['conditions'] if c['id']!='unsuccessful_completion'])
        frozen=read_json(root/'configs/facts_e3_protocol.json')
        self.assertEqual(digest(PROFILE),frozen['profile_hash'])


if __name__=='__main__':
    unittest.main()
