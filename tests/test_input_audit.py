import copy
from pathlib import Path
import tempfile
import unittest

from egc.input_audit import audit, review_flags
from egc.fact_experiment import prepare
from egc.io import read_json, read_rows, write_rows
from test_fact_experiment import case, PROFILE


class InputAuditTests(unittest.TestCase):
    def test_chinese_and_masked_sentence_hints(self):
        for text in ('建议本院对被告人判处二至三个月××。','建议判处被告人××至五年，并处罚金。',
                     '被告曾被判处有期徒刑三年，后释放。'):
            flags=review_flags(text)
            self.assertIn('possible_sentence_hint',{f['kind'] for f in flags})
            for f in flags:
                if f['start'] is not None:
                    self.assertEqual(text[f['start']:f['end']],f['quote'])

    def test_reasoning_multi_charge_and_ordinary_dates(self):
        for text in ('本院在量刑时将酌情予以考虑。','是累犯，应当从重处罚。','两罪应当数罪并罚。'):
            self.assertTrue(review_flags(text))
        self.assertFalse(review_flags('2015年3月被告购买物品三件，花费五元，随后在六月归还。'))

    def test_within_and_across_split_near_pairs_preserve_data(self):
        text='某被告在合成市甲村与同伙约定见面后一起乘车前往市场，对商户实施虚构交易，损失全部由测试数据定义。'
        a=case(1,text); b=case(2,text.replace('甲村','乙村')); c={**case(3,text),'split':'dev','sentence_months':999}
        named={'train':[a,b],'dev':[c]}; original=copy.deepcopy(named)
        report,queue,pairs=audit(named,focus=[b])
        self.assertEqual(named,original)
        self.assertEqual(len(pairs),3)
        self.assertEqual(report['cross_split_pairs'],2)
        self.assertEqual(report['focus']['flagged_rows'],1)
        self.assertTrue(all(p['labels_differ'] for p in pairs if p['cross_split']))
        self.assertTrue(all(r['review_decision'] is None for r in queue))

    def test_focus_must_be_unchanged_and_small_exact_duplicates_detected(self):
        a=case(1,'短句。'); b=case(2,'短句。')
        report,_,pairs=audit({'train':[a,b]})
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['method'],'normalized_exact')
        with self.assertRaises(ValueError):
            audit({'train':[a]},focus=[{**a,'sentence_months':999}])

    def test_new_cohort_selection_rejects_within_cohort_near_duplicate(self):
        known=case(1)
        text='某人在合成地区甲村虚构采购合同后获得测试资金，随后离开现场，所有当事人与细节均为离线测试而编造。'
        a=case(2,text); b=case(3,text.replace('甲村','乙村'))
        far=case(4,'另一份完全独立的虚构输入包含不同场景和交易方式，用于充足样本与去重边界检查。')
        robbery=case(5,'某地发生一起纯虚构抢劫案件，这是一条没有任何真实当事人的测试记录。','抢劫罪')
        with tempfile.TemporaryDirectory() as tmp:
            prepare([known,a,b,far,robbery],[known],{'items':[{'id':known['id'],'expected':{'returned_or_compensated':'supported'}}]},PROFILE,tmp,per_charge=2)
            rows=read_rows(Path(tmp)/'new.jsonl')
            self.assertEqual(len({r['id'] for r in rows}&{a['id'],b['id']}),1)
            self.assertIn(far['id'],{r['id'] for r in rows})
            self.assertEqual(read_json(Path(tmp)/'selection_manifest.json')['selection_version'],'cohort-within-near-v2')

    def test_cli_writes_review_only_and_does_not_overwrite(self):
        from egc.cli import parser, run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.jsonl'; write_rows(source,[case(1,'建议本院判处二至三个月××。')])
            before=source.read_bytes()
            args=parser().parse_args(['audit-inputs','--inputs',str(source),'--output-dir',str(root/'audit')])
            result=run(args)
            self.assertEqual(result['flagged_rows'],1); self.assertEqual(result['api_calls'],0)
            self.assertEqual(source.read_bytes(),before)
            self.assertTrue((root/'audit/overlap_pairs.json').exists())
            with self.assertRaises(ValueError): run(args)


if __name__=='__main__': unittest.main()
