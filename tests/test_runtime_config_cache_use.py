"""Per-cell config accessors must read the cache, not rebuild it.

Every accessor in the refine chunk called ``get_runtime_config(refresh=True)``,
which clears the ``lru_cache`` and re-reads every refine env var. Because *all* of
them refreshed, the cache never served a single hit: each access paid a full
rebuild, around 8 us against 42 ns.

``is_alignable_cell`` calls ``low_content_std_min_threshold`` once per cell, twice
per measurement, so a 10,000 cell grid spent about 0.16 s rebuilding a config that
had not changed.

Coarse callers in ``local_distortion_correction`` still refresh once per pass and
per tile, so a mid-process environment change is still picked up.
"""
from __future__ import annotations

import os
import unittest

from nornir_imageregistration.refine_shared.cell_roles import identity_zncc_min_threshold
from nornir_imageregistration.refine_shared.cell_validity import (
    low_content_std_min_threshold)
from nornir_imageregistration.refine_shared.discontinuity import (
    discontinuity_travel_multiplier, sharp_warps_enabled)
from nornir_imageregistration.refine_shared.finalize import use_legacy_finalize_gate
from nornir_imageregistration.refine_shared.pass_diagnostics import pass_diagnostics_enabled
from nornir_imageregistration.refine_shared.runtime_config import (
    _cached_config, get_runtime_config)

_ACCESSORS = (
    low_content_std_min_threshold,
    identity_zncc_min_threshold,
    sharp_warps_enabled,
    discontinuity_travel_multiplier,
    use_legacy_finalize_gate,
    lambda: pass_diagnostics_enabled(False),
)


class _ConfigCase(unittest.TestCase):
    """The config cache and the environment are both process-global."""

    _VARS = (
        'NORNIR_REFINE_LOW_CONTENT_STD_MIN',
        'NORNIR_REFINE_IDENTITY_ZNCC_MIN',
        'NORNIR_REFINE_SHARP_WARPS',
    )

    def setUp(self) -> None:
        self._saved = {name: os.environ.get(name) for name in self._VARS}
        get_runtime_config(refresh=True)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        get_runtime_config(refresh=True)


class TestAccessorsDoNotRebuildTheConfig(_ConfigCase):

    def test_repeated_reads_are_cache_hits(self):
        """The regression test: each call used to clear the cache and rebuild."""
        low_content_std_min_threshold()
        before = _cached_config.cache_info()

        for _ in range(10):
            low_content_std_min_threshold()

        after = _cached_config.cache_info()
        self.assertEqual(after.misses, before.misses, 'accessor rebuilt the config')
        self.assertEqual(after.hits, before.hits + 10)

    def test_every_hot_accessor_is_a_cache_hit(self):
        for accessor in _ACCESSORS:
            with self.subTest(accessor=getattr(accessor, '__name__', 'lambda')):
                accessor()
                before = _cached_config.cache_info()
                accessor()
                after = _cached_config.cache_info()

                self.assertEqual(after.misses, before.misses)
                self.assertEqual(after.hits, before.hits + 1)

    def test_config_identity_is_stable_across_accessor_calls(self):
        first = get_runtime_config()
        for accessor in _ACCESSORS:
            accessor()

        self.assertIs(get_runtime_config(), first,
                      'an accessor discarded the cached config object')


class TestRefreshStillPropagates(_ConfigCase):
    """Reading the cache is only safe if an explicit refresh still works."""

    def test_explicit_refresh_updates_the_accessor(self):
        os.environ['NORNIR_REFINE_LOW_CONTENT_STD_MIN'] = '0.25'
        get_runtime_config(refresh=True)

        self.assertAlmostEqual(low_content_std_min_threshold(), 0.25)

    def test_accessor_matches_the_config_field(self):
        os.environ['NORNIR_REFINE_IDENTITY_ZNCC_MIN'] = '0.75'
        config = get_runtime_config(refresh=True)

        self.assertAlmostEqual(identity_zncc_min_threshold(), config.identity_zncc_min)
        self.assertAlmostEqual(identity_zncc_min_threshold(), 0.75)

    def test_boolean_accessor_follows_a_refresh(self):
        os.environ['NORNIR_REFINE_SHARP_WARPS'] = '0'
        get_runtime_config(refresh=True)
        self.assertFalse(sharp_warps_enabled())

        os.environ['NORNIR_REFINE_SHARP_WARPS'] = '1'
        get_runtime_config(refresh=True)
        self.assertTrue(sharp_warps_enabled())


class TestCoarseCallersStillRefresh(_ConfigCase):
    """A refine pass must still pick up a late change without an explicit refresh."""

    def test_pass_entry_helper_refreshes_for_the_cell_accessors(self):
        from nornir_imageregistration import local_distortion_correction as ldc

        os.environ['NORNIR_REFINE_LOW_CONTENT_STD_MIN'] = '0.5'

        # Runs at the top of _RefinePointsForTwoImages, before the cell loop.
        ldc._use_batched_vertex_measurement()

        self.assertAlmostEqual(low_content_std_min_threshold(), 0.5)


if __name__ == '__main__':
    unittest.main()
