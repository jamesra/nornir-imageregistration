"""``NORNIR_REFINE_PHASE_TIMING`` must have exactly one source of truth.

``RefinePhaseTimer.__init__`` read the environment directly and stored the result,
and the process-global timer is constructed at import.  ``RefineRuntimeConfig``
re-reads the environment on ``get_runtime_config(refresh=True)``.

So a benchmark or test that exported the variable after importing
``nornir_imageregistration`` ended up with the config reporting timing enabled
while the timer stayed disabled and every bucket came back empty.  That is why
``scripts/microbench_grid_refine.py`` has to pin its flags *before* the nornir
imports.

``enabled`` now defers to the runtime config unless explicitly overridden.
"""
from __future__ import annotations

import os
import unittest

from nornir_imageregistration.refine_shared.phase_timer import (
    RefinePhaseTimer, get_phase_timer)
from nornir_imageregistration.refine_shared.runtime_config import get_runtime_config

_ENV = 'NORNIR_REFINE_PHASE_TIMING'


class _EnvCase(unittest.TestCase):
    """Restores the flag and the config cache, both process-global."""

    def setUp(self) -> None:
        self._saved = os.environ.get(_ENV)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = self._saved
        get_runtime_config(refresh=True)

    @staticmethod
    def _set_flag(value: str | None):
        if value is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = value
        return get_runtime_config(refresh=True)


class TestTimerFollowsTheRuntimeConfig(_EnvCase):

    def test_late_enable_is_picked_up(self):
        """The reported bug: env set after import left the timer latched off."""
        timer = RefinePhaseTimer()
        self._set_flag('0')
        self.assertFalse(timer.enabled)

        config = self._set_flag('1')

        self.assertTrue(config.phase_timing)
        self.assertTrue(timer.enabled, 'timer must not stay latched at its import value')

    def test_late_disable_is_picked_up(self):
        timer = RefinePhaseTimer()
        self._set_flag('1')
        self.assertTrue(timer.enabled)

        self._set_flag('0')

        self.assertFalse(timer.enabled)

    def test_buckets_record_after_a_late_enable(self):
        """Agreement is not enough; the sections have to actually record."""
        timer = RefinePhaseTimer()
        self._set_flag('0')
        with timer.section('fft'):
            pass
        self.assertEqual(timer.snapshot(), {})

        self._set_flag('1')
        with timer.section('fft'):
            pass

        self.assertIn('fft', timer.snapshot())
        self.assertGreater(timer.totals['fft'], 0.0)
        self.assertEqual(timer.counts['fft'], 1)

    def test_process_global_timer_also_follows(self):
        """The global instance is the one built at import, so it is the real case."""
        timer = get_phase_timer()
        self._set_flag('1')
        try:
            self.assertTrue(timer.enabled)
        finally:
            self._set_flag('0')
        self.assertFalse(timer.enabled)

    def test_agrees_with_config_across_flag_values(self):
        for value in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off',
                      '', 'garbage', 'TRUE', 'Off', ' 1 ', None):
            with self.subTest(value=value):
                config = self._set_flag(value)
                timer = RefinePhaseTimer()

                self.assertEqual(timer.enabled, config.phase_timing)


class TestLegacyTruthTableIsPreserved(_EnvCase):
    """Unifying the two readers must not change which values mean "on"."""

    @staticmethod
    def _legacy_enabled(value: str | None) -> bool:
        """The parsing RefinePhaseTimer.__init__ used to do."""
        flag = (value if value is not None else '0').strip().lower()
        return flag not in ('', '0', 'false', 'no', 'off')

    def test_same_verdict_as_the_old_parsing(self):
        for value in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off',
                      '', 'garbage', 'TRUE', 'Off', ' 1 ', None):
            with self.subTest(value=value):
                self._set_flag(value)
                timer = RefinePhaseTimer()

                self.assertEqual(timer.enabled, self._legacy_enabled(value))


class TestExplicitOverride(_EnvCase):
    """Tests pass ``enabled=True`` directly and must keep winning."""

    def test_override_true_beats_a_disabled_config(self):
        config = self._set_flag('0')
        timer = RefinePhaseTimer(enabled=True)

        self.assertFalse(config.phase_timing)
        self.assertTrue(timer.enabled)
        with timer.section('fft'):
            pass
        self.assertIn('fft', timer.snapshot())

    def test_override_false_beats_an_enabled_config(self):
        self._set_flag('1')
        timer = RefinePhaseTimer(enabled=False)

        self.assertFalse(timer.enabled)
        with timer.section('fft'):
            pass
        self.assertEqual(timer.snapshot(), {})

    def test_assignment_pins_the_flag(self):
        self._set_flag('0')
        timer = RefinePhaseTimer()
        timer.enabled = True

        self.assertTrue(timer.enabled)
        self._set_flag('0')
        self.assertTrue(timer.enabled, 'an explicit pin must survive a config refresh')

    def test_assigning_none_restores_config_following(self):
        self._set_flag('0')
        timer = RefinePhaseTimer()
        timer.enabled = True
        timer.enabled = None

        self.assertIsInstance(timer.enabled, bool,
                              'enabled must read as a bool, never as None')
        self.assertFalse(timer.enabled)
        self._set_flag('1')
        self.assertTrue(timer.enabled)


class TestSectionWallIsUnconditional(_EnvCase):
    """section_wall records regardless of the flag; that contract is unchanged."""

    def test_records_when_disabled(self):
        self._set_flag('0')
        timer = RefinePhaseTimer()

        with timer.section_wall('finalize'):
            pass

        self.assertIn('finalize', timer.snapshot())
        self.assertEqual(timer.counts['finalize'], 1)


if __name__ == '__main__':
    unittest.main()
