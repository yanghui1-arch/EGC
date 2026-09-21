import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from egc.b0 import (ARMS, INPUT_FILES, RESULT_FILES, build, checked_predictions, collect,
                    encoded, evaluate, pack_files, prepare, read_archive, server_run,
                    validate_inputs)
from egc.e4_inventory import TRAIN_MEMBER, source_metadata
from egc.io import digest, read_json, read_rows, write_json, write_rows


class B0Tests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(id=str(i), source_id=f'{TRAIN_MEMBER}:{i}', split='dev',
                     charge=charge, facts=facts, sentence_months=months,
                     opinion='SECRET_OPINION', relevant_articles_label=[987654321])
                for i, charge, facts, months in [
                    (1, '诈骗罪', '甲被抓获，乙自行到案。这是合成案情。', 24),
                    (2, '抢劫罪', '匿名人X实施虚构行为。这是合成案情。', 36)]]
        train = root/'dev.jsonl'; archive = root/'source.zip'
        write_rows(train, rows)
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(TRAIN_MEMBER, '\n'.join(json.dumps({'fact':r['facts'],
                'meta':{'criminals':['甲' if r['id']=='1' else '乙'],
                        'term_of_imprisonment':{'imprisonment':r['sentence_months']}}}) for r in rows))
        protocol = {'version':'b0-target-input-diagnostic-v1', 'dev_hash':digest(rows),
                    'dev_rows':2, 'charges':{'诈骗罪':1, '抢劫罪':1},
                    'model':'/mnt/yanghui/models/Qwen/Qwen3-4B',
                    'decoding':{'batch_size':2,'max_model_len':16384,'max_new_tokens':1024,
                                'tensor_parallel':1,'gpu_memory':0.85,'temperature':0.0,'seed':42}}
        return rows, train, archive, protocol

    def fake_results(self, jobs, protocol, run):
        for arm in ARMS:
            settings = {k:v for k,v in protocol['decoding'].items() if k!='gpu_memory'}
            settings.update(model=protocol['model'], adapter=None, adapter_config=None,
                            jobs_hash=digest(jobs[arm]), model_config={'model_type':'synthetic'},
                            weight_inventory=[], rendered_prompts_hash=arm)
            key=digest(settings)
            manifest={'settings':settings,'generation_key':key}
            preds=[dict(id=j['id'], prompt_hash=j['prompt_hash'], generation_key=key,
                        text=json.dumps({'reasoning':'虚构测试理由', 'sentence_months':int(j['id'])*12+12}),
                        finish_reason='stop',prompt_tokens=100,output_tokens=30) for j in jobs[arm]]
            write_rows(run/f'predictions_{arm}.jsonl', preds)
            write_json(run/f'predictions_{arm}.jsonl.manifest.json', manifest)
        write_json(run/'execution.json',{'arms':{},'test_only':True})

    def test_pack_label_isolation_mask_pairing_and_no_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows, dev, source, protocol=self.fixture(root)
            before=dev.read_bytes()
            result=prepare(dev,source,root/'prepared',protocol)
            self.assertEqual(result['generations'],4); self.assertEqual(result['target_available'],1)
            self.assertEqual(dev.read_bytes(),before)
            data=read_archive(root/'prepared/b0_jobs.zip',INPUT_FILES)
            manifest,masks,jobs=validate_inputs(data,protocol)
            self.assertEqual([m['available'] for m in masks],[True,False])
            for payload in data.values():
                self.assertNotIn(b'SECRET_OPINION',payload)
                self.assertNotIn(b'987654321',payload)
            with zipfile.ZipFile(root/'prepared/b0_jobs.zip') as z:
                self.assertNotIn('references.local.jsonl',z.namelist())
            first=json.loads(jobs['target'][0]['messages'][1]['content'])
            self.assertEqual(first.pop('target_person'),'甲')
            self.assertEqual(first,json.loads(jobs['original'][0]['messages'][1]['content']))
            self.assertEqual(jobs['original'][1]['messages'],jobs['target'][1]['messages'])
            changed=copy.deepcopy(rows)
            for row in changed: row['sentence_months']=999; row['opinion']='CHANGED'
            alt=dict(protocol,dev_hash=digest(changed))
            other=build(changed,source_metadata(rows,source),alt)
            self.assertEqual(jobs,other[0])
            with self.assertRaisesRegex(ValueError,'fresh'): prepare(dev,source,root/'prepared',protocol)

    def test_reject_wrong_split_snapshot_or_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows,dev,source,protocol=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'snapshot'):
                prepare(dev,source,root/'out',dict(protocol,dev_hash='wrong'))
            rows[0]['split']='test'; write_rows(dev,rows)
            with self.assertRaisesRegex(ValueError,'snapshot'):
                prepare(dev,source,root/'out',dict(protocol,dev_hash=digest(rows)))
            rows[0]['split']='dev'; rows[0]['facts']='changed'; write_rows(dev,rows)
            with self.assertRaisesRegex(ValueError,'Source facts differ'):
                prepare(dev,source,root/'out',dict(protocol,dev_hash=digest(rows)))
            self.assertFalse((root/'out').exists())

    def test_reject_tampered_archive_and_extra_label_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,dev,source,p=self.fixture(root)
            prepare(dev,source,root/'p',p)
            data=read_archive(root/'p/b0_jobs.zip',INPUT_FILES)
            bad=dict(data); bad['../escape']=b'bad'; pack_files(root/'bad.zip',bad)
            with self.assertRaisesRegex(ValueError,'archive members'): read_archive(root/'bad.zip',INPUT_FILES)
            with zipfile.ZipFile(root/'mismatch.zip','w') as z:
                for k,v in data.items(): z.writestr(k,v)
                z.writestr('checksums.json','{}')
            with self.assertRaisesRegex(ValueError,'checksum'): read_archive(root/'mismatch.zip',INPUT_FILES)
            job=json.loads(data['original.jobs.jsonl'].splitlines()[0]); job['reference']=999
            lines=[job]+[json.loads(x) for x in data['original.jobs.jsonl'].splitlines()[1:]]
            data['original.jobs.jsonl']=b''.join(encoded(j).replace(b'\n',b'')+b'\n' for j in lines)
            m=json.loads(data['manifest.json']); m['jobs_hash']['original']=digest(lines); data['manifest.json']=encoded(m)
            with self.assertRaisesRegex(ValueError,'Unexpected job fields'): validate_inputs(data,p)

    def test_full_result_cycle_failure_coverage_and_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,dev,source,p=self.fixture(root)
            prepare(dev,source,root/'p',p)
            data=read_archive(root/'p/b0_jobs.zip',INPUT_FILES)
            _,_,jobs=validate_inputs(data,p)
            run=root/'run'; run.mkdir(); self.fake_results(jobs,p,run)
            preds=read_rows(run/'predictions_target.jsonl'); preds[1]['finish_reason']='length'
            write_rows(run/'predictions_target.jsonl',preds)
            result=collect(data,run,p)
            self.assertTrue(result['generation_complete']); self.assertEqual(result['valid_format']['target'],1)
            metrics=evaluate(root/'p',run/'b0_results.zip',root/'metrics.json',p)
            self.assertIsNone(metrics['scopes']['all_cases']['delta_mae_target_minus_original'])
            self.assertIsNone(metrics['scopes']['all_cases']['arms']['target']['mae_months_full'])
            self.assertEqual(metrics['scopes']['target_available']['n'],1)
            self.assertEqual(metrics['scopes']['target_available']['delta_mae_target_minus_original'],0)
            self.assertEqual(metrics['scopes']['all_cases']['arms']['original']['mae_months_full'],0)
            with self.assertRaisesRegex(ValueError,'new metrics'): evaluate(root/'p',run/'b0_results.zip',root/'metrics.json',p)

    def test_reject_different_prediction_identity_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,dev,source,p=self.fixture(root)
            prepare(dev,source,root/'p',p)
            data=read_archive(root/'p/b0_jobs.zip',INPUT_FILES); _,_,jobs=validate_inputs(data,p)
            run=root/'run';run.mkdir();self.fake_results(jobs,p,run)
            raw={n:(run/n).read_bytes() for n in RESULT_FILES if (run/n).exists()}
            rows=read_rows(run/'predictions_target.jsonl'); rows[0]['prompt_hash']='wrong'
            raw['predictions_target.jsonl']=b'\n'.join(json.dumps(r).encode() for r in rows)
            with self.assertRaisesRegex(ValueError,'match job'): checked_predictions(raw,jobs,p)
            raw['predictions_target.jsonl']=(run/'predictions_target.jsonl').read_bytes()
            m=json.loads(raw['predictions_target.jsonl.manifest.json']);m['settings']['model']='other'
            raw['predictions_target.jsonl.manifest.json']=encoded(m)
            with self.assertRaisesRegex(ValueError,'identity/model'): checked_predictions(raw,jobs,p)

    def test_orchestrator_with_fake_child_processes_never_loads_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,dev,source,p=self.fixture(root)
            prepare(dev,source,root/'p',p)
            data=read_archive(root/'p/b0_jobs.zip',INPUT_FILES); _,_,jobs=validate_inputs(data,p)
            run=root/'run'; calls=[]
            def fake(command,**kwargs):
                if command[0]=='git': return subprocess.CompletedProcess(command,0,stdout='test-commit\n')
                self.assertIn('infer',command); self.assertNotIn('train',command)
                self.assertEqual(command[command.index('--max-model-len')+1],'16384')
                calls.append(command)
                self.fake_results(jobs,p,run)
                return subprocess.CompletedProcess(command,0)
            with patch('egc.b0.sys.platform','linux'),patch('egc.b0.subprocess.run',side_effect=fake):
                result=server_run(root/'p/b0_jobs.zip',run,p)
            self.assertEqual(len(calls),2)
            self.assertTrue(Path(result['results_zip']).is_file())
            self.assertEqual(set(read_archive(result['results_zip'],RESULT_FILES)),RESULT_FILES)


if __name__=='__main__': unittest.main()
