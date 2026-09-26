import unittest

from egc.diagnose_qwen_stop import compare_rows


class StopDiagnosticTests(unittest.TestCase):
    def test_exact_alias_diff_and_invalid_rows(self):
        rows = {151645: [0.25, -0.5], 123: [0.25, -0.5], 456: [0.25, -0.25]}
        report = compare_rows(rows, 151645)
        self.assertTrue(report["123"]["exactly_equals_im_end"])
        self.assertFalse(report["456"]["exactly_equals_im_end"])
        self.assertEqual(report["456"]["max_absolute_difference"], 0.25)
        for invalid in ([0.25], [float("nan"), 0], []):
            with self.assertRaises(ValueError):
                compare_rows({151645: rows[151645], 123: invalid}, 151645)
