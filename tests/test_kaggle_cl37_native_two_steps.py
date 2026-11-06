"""CL-37 offline tests: no GGUF download, no GPU consumption."""
import ast
import re
import unittest

from scripts import kaggle_cl37_native_two_steps as cl37


class Cl37Tests(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(re.sub(r"[^a-z0-9]+", "-", cl37.TITLE.lower()).strip("-"), cl37.SLUG)

    def test_two_native_evaluations_and_feedback(self):
        code = cl37._NATIVE_TWO_STEPS_BLOCK
        self.assertEqual(code.count('for index in range(8):'), 4)
        self.assertIn('first_native_update = latent.float() + first_delta * native_output.float()', code)
        self.assertIn('second_native_input = first_native_update.to(dtype=latent.dtype)', code)
        self.assertIn('root, second_native_input, None, None)', code)
        self.assertIn('linked_ts = torch.full((1, 4), native_sigmas[1]', code)
        self.assertIn('v_timestep=linked_prepared[0]', code)
        self.assertIn('v_prompt_timestep=linked_prepared[3]', code)
        self.assertIn('root, linked_vx, linked_out_embed, None, **linked_meta)', code)
        self.assertIn('"native_adaln_both_passes": True', code)
        self.assertIn('"real_ltx_scheduler": False', code)
        self.assertLess(code.index('first_native_update ='), code.index('second_native_input ='))
        self.assertIn('if peak_rss > 12_000_000_000:', cl37.KERNEL_SOURCE)
        self.assertIn('child.kill()', cl37.KERNEL_SOURCE)
        ast.parse(code)

    def test_missing_report_rejected(self):
        with self.assertRaises(ValueError):
            cl37.validate_report({}, "sha", "run")


if __name__ == '__main__':
    unittest.main()
