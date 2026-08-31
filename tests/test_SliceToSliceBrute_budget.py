"""The ``slow`` exclusions in test_SliceToSliceBrute.py must stay measured and narrow.

Review #229 reported that file as hanging and, on that basis, it was being skipped by
hand during verification. It does not hang: sampling the stuck thread showed the pool
queue draining steadily (179 -> 11 jobs over 900s), so the parent was simply asleep in
its 0.5s poll waiting on real work. The work is just enormous. ``AngleSearchRange=None``
runs a full angle sweep at each of 11 scale candidates, and the inputs pad to an
8192x8192 float32 frame (256 MiB) at ~10.5s per angle serially -- hundreds of minutes of
FFT for one test.

Measuring all 33 tests individually (150s each) put 9 over budget and 24 under, with the
24 totalling ~800s. So the wholesale exclusion was costing 24 usable tests to avoid 9,
and the marker now carries that split explicitly.

These tests guard the two ways that split can rot:

* the marked set drifting from the measured set, in either direction -- marking a cheap
  test hides it for no reason, and leaving an expensive one unmarked puts the killed-run
  churn back;
* the marked set growing until ``-m "not slow"`` no longer exercises the CPU scoring path
  at all. Every over-budget test is a numpy full-sweep variant, so deselecting them could
  plausibly have left only GPU coverage. It does not, and that has to stay true.
"""

from __future__ import annotations

import configparser
import pathlib
import unittest

import test_SliceToSliceBrute as brute

# Measured on 2026-08-30, one subprocess per test, 150s budget each. Recorded here rather
# than re-measured because re-measuring costs the very hours this marker exists to avoid.
OVER_BUDGET = frozenset({
    'TestStosBrute::testStosBrute_SingleThread',
    'TestStosBrute::testStosBrute_MultiThread',
    'TestStosBrute::testStosBrute_Cluster',
    'TestStosBrute::testStosBruteWithFlip_SingleThread',
    'TestStosBrute::testStosBruteWithFlip_MultiThread',
    'TestStosBrute::testStosBruteWithFlip_Cluster',
    'TestStosBruteWithMask::testStosBruteWithMask_MultiThread',
    'TestStosBruteWithMask::testStosBruteScaleMismatchWithMask',
    'TestStosBruteToSameImage::testSameTEMImage_MultiThread',
})

# Under budget and exercising numpy (not cupy) scoring, so ``-m "not slow"`` keeps real
# CPU coverage of the brute path.
CPU_COVERAGE_UNDER_BUDGET = frozenset({
    'TestStosBruteToSameImage::testSameTEMImageFast_SingleThread',
    'TestStosBruteToSameImage::testSameTEMImageFast_MultiThread',
    'TestStosBruteWithMask::testStosBruteExecuteWithMask',
})


def _test_methods() -> dict[str, object]:
    """Every ``test*`` method in the module, keyed ``Class::method``."""
    found: dict[str, object] = {}
    for class_name in dir(brute):
        candidate = getattr(brute, class_name)
        if not isinstance(candidate, type) or not issubclass(candidate, unittest.TestCase):
            continue
        for attr in dir(candidate):
            if attr.startswith('test'):
                method = getattr(candidate, attr, None)
                if callable(method):
                    found[f'{class_name}::{attr}'] = method
    return found


def _is_slow(method: object) -> bool:
    return any(mark.name == 'slow' for mark in getattr(method, 'pytestmark', []))


class TestTheMarkedSetMatchesTheMeasuredSet(unittest.TestCase):

    def setUp(self):
        self.methods = _test_methods()
        self.marked = {name for name, method in self.methods.items() if _is_slow(method)}

    def test_every_over_budget_test_is_marked(self):
        missing = OVER_BUDGET - self.marked
        self.assertEqual(set(), missing,
                         'measured over 150s but not marked slow, so a routine run will '
                         f'stall on it again: {sorted(missing)}')

    def test_no_test_is_marked_without_being_measured(self):
        extra = self.marked - OVER_BUDGET
        self.assertEqual(set(), extra,
                         'marked slow but not in the measured over-budget set; marking on '
                         f'suspicion silently drops coverage: {sorted(extra)}')

    def test_the_measured_names_still_exist(self):
        unknown = OVER_BUDGET - set(self.methods)
        self.assertEqual(set(), unknown,
                         'the measurement refers to tests that no longer exist; renaming a '
                         f'test must update the record: {sorted(unknown)}')

    def test_the_exclusion_is_a_minority_of_the_file(self):
        # The point of #229 was that excluding the whole file cost 24 usable tests. If the
        # marked set ever reaches most of the file, the narrowing has been undone.
        self.assertLess(len(self.marked), len(self.methods) / 2,
                        f'{len(self.marked)} of {len(self.methods)} tests are marked slow; '
                        'this is close to excluding the file again')


class TestDeselectingSlowKeepsCpuCoverage(unittest.TestCase):
    """``-m "not slow"`` must still exercise numpy scoring, not only the GPU mirrors."""

    def setUp(self):
        self.methods = _test_methods()

    def test_the_cpu_tests_named_here_exist(self):
        unknown = CPU_COVERAGE_UNDER_BUDGET - set(self.methods)
        self.assertEqual(set(), unknown, f'unknown test names: {sorted(unknown)}')

    def test_the_cpu_tests_named_here_are_not_marked_slow(self):
        for name in sorted(CPU_COVERAGE_UNDER_BUDGET):
            with self.subTest(name=name):
                self.assertFalse(
                    _is_slow(self.methods[name]),
                    f'{name} is the CPU coverage that survives -m "not slow"; marking it '
                    'leaves the numpy brute path untested in routine runs')

    def test_something_unmarked_still_covers_the_brute_path(self):
        unmarked = {name for name, method in self.methods.items() if not _is_slow(method)}
        self.assertTrue(unmarked & CPU_COVERAGE_UNDER_BUDGET)


class TestTheMarkerIsDeclared(unittest.TestCase):
    """An undeclared marker filters nothing and only warns, which is how it went unnoticed.

    ``slow`` was already applied in ``test_refine_single_pass.py`` while absent from
    ``pytest.ini``, so it raised PytestUnknownMarkWarning and ``-m "not slow"`` was not
    doing what the marker implied.

    Both config files are checked because the submodule's ``[tool.pytest.ini_options]``
    shadows the umbrella ini when pytest runs from inside the submodule -- which is how
    per-file verification is run, so declaring it only upstream fixes the wrong case.
    """

    @property
    def _repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[2]

    @property
    def _submodule_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[1]

    def _umbrella_markers(self) -> list[str]:
        parser = configparser.ConfigParser()
        parser.read(self._repo_root / 'pytest.ini', encoding='utf-8')
        raw = parser.get('pytest', 'markers', fallback='')
        return [line.split(':', 1)[0].strip() for line in raw.splitlines() if line.strip()]

    def _submodule_markers(self) -> list[str]:
        text = (self._submodule_root / 'pyproject.toml').read_text(encoding='utf-8')
        section = text.partition('[tool.pytest.ini_options]')[2]
        # Stop at the next table so a "markers" key elsewhere cannot satisfy this.
        section = section.partition('\n[')[0]
        names: list[str] = []
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith('"') and ':' in stripped:
                names.append(stripped.lstrip('"').split(':', 1)[0].strip())
        return names

    def test_the_umbrella_ini_exists(self):
        ini = self._repo_root / 'pytest.ini'
        self.assertTrue(ini.is_file(), f'expected the umbrella pytest.ini at {ini}')

    def test_slow_is_declared_in_the_umbrella_ini(self):
        self.assertIn('slow', self._umbrella_markers(),
                      'the slow marker must be declared or -m "not slow" silently '
                      'deselects nothing')

    def test_slow_is_declared_in_the_submodule_config(self):
        self.assertIn('slow', self._submodule_markers(),
                      'the submodule pytest config shadows the umbrella ini for runs '
                      'started inside the submodule, so slow must be declared here too')

    def test_graphical_is_still_declared_upstream(self):
        self.assertIn('graphical', self._umbrella_markers())


if __name__ == '__main__':
    unittest.main()
