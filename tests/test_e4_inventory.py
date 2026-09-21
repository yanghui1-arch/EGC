import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from egc.e4_inventory import TRAIN_MEMBER, cue_spans, inventory
from egc.io import digest, read_json, read_rows, write_rows

RULES = Path(__file__).resolve().parents[1] / 'configs/rules_e4_draft.json'


class E4InventoryTests(unittest.TestCase):
    def fixtures(self, root):
        rows = [dict(id=str(i), source_id=f'{TRAIN_MEMBER}:{i}', split='train',
                     charge=charge, facts=text, sentence_months=777,
                     relevant_articles_label=[999], opinion='SECRET_OPINION')
                for i, charge, text in [
                    (1, '诈骗罪', '甲未退赔，并非翻供；2014年1月到案。'),
                    (2, '抢劫罪', '甲被抓获，乙投案。随后有人乘出租车逃走。')]]
        train = root / 'train.jsonl'
        archive = root / 'source.zip'
        write_rows(train, rows)
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(TRAIN_MEMBER, '\n'.join(json.dumps({
                'fact': r['facts'], 'meta': {'criminals': ['甲'],
                'term_of_imprisonment': {'imprisonment': 777},
                'accusation': ['SECRET_ACCUSATION']}}) for r in rows))
        return train, archive, rows

    def test_offsets_and_negation_are_hints_only(self):
        text = '甲未退赔，并非翻供。乙逃跑，甲没有逃跑。'
        spans = cue_spans(text)
        self.assertEqual(sum(s['cue'] == 'escape' for s in spans), 2)
        for span in spans:
            self.assertEqual(text[span['start']:span['end']], span['quote'])
            self.assertNotIn('state', span)

    def test_label_blind_review_source_binding_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, archive, rows = self.fixtures(root)
            before = train.read_bytes()
            report = inventory(train, archive, RULES, root/'out', digest(rows))
            self.assertEqual(report['selected_rows'], 2)
            self.assertEqual(report['reviewed_conditions'], 0)
            self.assertEqual(report['api_calls'], 0)
            self.assertEqual(train.read_bytes(), before)
            content = (root/'out/review_cases.jsonl').read_text(encoding='utf-8')
            for forbidden in ('sentence_months', 'relevant_articles_label', 'SECRET_', '777'):
                self.assertNotIn(forbidden, content)
            reviews = read_rows(root/'out/review_cases.jsonl')
            for row in reviews:
                self.assertEqual(row['source']['target_names_from_meta'], ['甲'])
                self.assertIsNone(row['source']['target_binding_review'])
                self.assertTrue(all(c['reviewed_state'] is None for c in row['conditions_to_review']))
            self.assertEqual(read_json(root/'out/report.json')['review_hash'], digest(reviews))
            # Different numeric/article/outcome labels cannot affect review contents or cues.
            changed = copy.deepcopy(rows)
            for row in changed:
                row.update(sentence_months=13, relevant_articles_label=[42], opinion='different')
            write_rows(train, changed)
            inventory(train, archive, RULES, root/'out2')
            self.assertEqual(reviews, read_rows(root/'out2/review_cases.jsonl'))

    def test_reject_dev_or_wrong_snapshot_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, archive, rows = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                inventory(train, archive, RULES, root/'out', 'wrong')
            rows[1]['split'] = 'dev'; write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, 'never dev/test'):
                inventory(train, archive, RULES, root/'out')
            self.assertFalse((root/'out').exists())

    def test_source_mismatch_missing_line_and_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, archive, rows = self.fixtures(root)
            inventory(train, archive, RULES, root/'out')
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                inventory(train, archive, RULES, root/'out')
            rows[0]['facts'] = 'Changed source'
            write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, 'Source facts differ'):
                inventory(train, archive, RULES, root/'bad')
            self.assertFalse((root/'bad').exists())
            train, archive, rows = self.fixtures(root)
            rows[1]['source_id'] = f'{TRAIN_MEMBER}:999'
            write_rows(train, rows)
            with self.assertRaisesRegex(ValueError, 'not found'):
                inventory(train, archive, RULES, root/'missing')
            self.assertFalse((root/'missing').exists())


if __name__ == '__main__':
    unittest.main()
