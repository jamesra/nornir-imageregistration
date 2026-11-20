"""The .stos registration scripts refuse to run without both Assemble masks (James, 2026-11-20).

They used to fall back to an unmasked registration, without a log line, when the input .stos
named no masks and no -fm/-wm was given.
"""
from __future__ import annotations

import unittest
from unittest import mock

import nornir_imageregistration.scripts.nornir_rotate_translate as rotate_translate
import nornir_imageregistration.scripts.nornir_stos_grid_refinement as stos_grid_refinement
from nornir_imageregistration.scripts._cli_use_error import USE_ERROR_EXIT_CODE


class TestStosScriptsRequireMasks(unittest.TestCase):

    def test_rotate_translate_exits_without_a_mask(self) -> None:
        with mock.patch.object(rotate_translate, 'StosOverrideArgs') as args, \
                mock.patch.object(rotate_translate.sb, 'SliceToSliceRigidRegistration') as register:
            args.return_value.ControlMask = 'fixed_mask.png'
            args.return_value.WarpedMask = None
            with self.assertRaises(SystemExit) as stop:
                rotate_translate.Execute(['-f', 'a.png', '-w', 'b.png', '-o', 'out.stos'])
        self.assertEqual(USE_ERROR_EXIT_CODE, stop.exception.code)
        register.assert_not_called()

    def test_stos_grid_refinement_exits_without_a_mask(self) -> None:
        with mock.patch.object(stos_grid_refinement, 'StosOverrideArgs') as args, \
                mock.patch.object(stos_grid_refinement.nornir_imageregistration.local_distortion_correction,
                                  'RefineStosFile') as refine:
            merged = args.return_value.MergeStosAndArgs.return_value
            merged.ControlMaskFullPath = None
            merged.MappedMaskFullPath = None
            with self.assertRaises(SystemExit) as stop:
                stos_grid_refinement.Execute(['-i', 'a.stos', '-o', 'out.stos'])
        self.assertEqual(USE_ERROR_EXIT_CODE, stop.exception.code)
        refine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
