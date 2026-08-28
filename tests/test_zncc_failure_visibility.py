"""A ZNCC that could not be computed is not a ZNCC of zero.

``_compute_zncc_for_candidates`` had two silent handlers:

* the batched path was wrapped in ``except Exception: pass``, so a real defect in
  batched ZNCC degraded every pass to the slow per-candidate path with no signal;
* the per-candidate path recorded ``scores[key] = 0.0`` on failure.

Zero is a legitimate ZNCC meaning "does not correlate", and it sits below
``identity_zncc_min``, so an infrastructure failure was indistinguishable from a
measured verdict.  ``classify_roles`` counted it in ``n_zncc_eval`` as though it
had been evaluated, and ``pass_diagnostics`` recorded a score that was never
measured.

The role outcome is unchanged -- ``classify_roles`` fails closed on a missing key
just as it does on a sub-threshold one -- so this is about not fabricating
evidence, and about being able to see the failure at all.
"""
from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
from nornir_imageregistration import local_distortion_correction as ldc
from nornir_imageregistration.refine_shared.cell_roles import Role, classify_roles
from nornir_shared import prettyoutput

_CELL = np.array((64, 64), dtype=np.int64)
_POINTS = np.array([(64.0, 64.0), (96.0, 64.0), (64.0, 96.0)])


def _records():
    return [
        nornir_imageregistration.EnhancedAlignmentRecord(
            ID=(i, 0),
            TargetPoint=_POINTS[i],
            SourcePoint=_POINTS[i],
            peak=np.array((0.0, 0.0)),
            weight=10.0,
            angle=0.0,
            flipped_ud=False,
            peak_ratio=5.0)
        for i in range(len(_POINTS))]


def _candidate_ids():
    return {(i, 0) for i in range(len(_POINTS))}


def _settings():
    image = np.random.default_rng(3).random((256, 256)).astype(np.float32)
    return nornir_imageregistration.settings.GridRefinement.CreateWithUnproccessedImages(
        target_image=image.copy(),
        source_image=image.copy(),
        cell_size=_CELL,
        single_thread_processing=True)


def _identity():
    return nornir_imageregistration.transforms.RigidTranslation(
        target_offset=np.array((0.0, 0.0)))


def _boom(*args, **kwargs):
    raise RuntimeError('simulated ZNCC failure')


class _ZnccCase(unittest.TestCase):
    """Runs the scorer with chosen internals broken, capturing logged errors."""

    def _score(self, **patches):
        logged: list[str] = []
        real_logerr = prettyoutput.LogErr

        def capture(message=None, calling_func_name=None):
            logged.append(str(message))

        saved = {name: getattr(ldc, name) for name in patches}
        prettyoutput.LogErr = capture
        for name, replacement in patches.items():
            setattr(ldc, name, replacement)
        try:
            scores = ldc._compute_zncc_for_candidates(
                _records(), _candidate_ids(), _identity(), _settings())
        finally:
            for name, original in saved.items():
                setattr(ldc, name, original)
            prettyoutput.LogErr = real_logerr
        return scores, logged


class TestHealthyPathIsUnchanged(_ZnccCase):

    def test_scores_every_candidate_without_logging(self):
        scores, logged = self._score()

        self.assertEqual(set(scores.keys()), _candidate_ids())
        self.assertEqual(logged, [], 'a working path must not log errors')

    def test_identical_images_score_near_one(self):
        scores, _ = self._score()

        for key, score in scores.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(score, 1.0, places=5)


class TestBatchedFailureIsVisibleAndRecovers(_ZnccCase):

    def test_failure_is_logged(self):
        _scores, logged = self._score(_masked_zncc_stack=_boom)

        self.assertEqual(len(logged), 1)
        self.assertIn('Batched ZNCC failed', logged[0])

    def test_per_candidate_fallback_still_produces_scores(self):
        scores, _logged = self._score(_masked_zncc_stack=_boom)

        self.assertEqual(set(scores.keys()), _candidate_ids())
        for score in scores.values():
            self.assertAlmostEqual(score, 1.0, places=5)


class TestUnmeasurableCandidateIsNotScoredZero(_ZnccCase):

    def test_key_is_absent_rather_than_zero(self):
        scores, _logged = self._score(
            _masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)

        self.assertEqual(scores, {})
        for key in _candidate_ids():
            self.assertNotIn(key, scores)

    def test_every_failure_is_logged(self):
        _scores, logged = self._score(
            _masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)

        # One for the batched attempt, one per candidate.
        self.assertEqual(len(logged), 1 + len(_POINTS))
        per_candidate = [m for m in logged if m.startswith('ZNCC scoring failed')]
        self.assertEqual(len(per_candidate), len(_POINTS))
        for key in _candidate_ids():
            self.assertTrue(any(str(key) in m for m in per_candidate),
                            f'no log names candidate {key}')

    def test_mass_failure_is_summarised_rather_than_logged_per_cell(self):
        """A systematic failure must not emit one line per cell on a large grid."""
        original_limit = ldc._ZNCC_FAILURE_LOG_LIMIT
        ldc._ZNCC_FAILURE_LOG_LIMIT = 1
        try:
            _scores, logged = self._score(
                _masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)
        finally:
            ldc._ZNCC_FAILURE_LOG_LIMIT = original_limit

        detailed = [m for m in logged if m.startswith('ZNCC scoring failed for lock candidate')]
        summary = [m for m in logged if 'of 3 lock candidates' in m]

        self.assertEqual(len(detailed), 1)
        self.assertEqual(len(summary), 1)
        self.assertIn('cannot lock', summary[0])

    def test_a_genuine_zero_score_is_still_recorded(self):
        """Zero must keep its meaning; only unmeasurable cells are omitted."""
        scores, logged = self._score(
            _masked_zncc_stack=lambda *a, **kw: np.zeros(len(_POINTS), dtype=np.float64))

        self.assertEqual(set(scores.keys()), _candidate_ids())
        for score in scores.values():
            self.assertEqual(score, 0.0)
        self.assertEqual(logged, [])


class TestConsumerStillFailsClosed(_ZnccCase):
    """Omitting the key must not let an unmeasurable cell lock."""

    def _classify(self, scores):
        return classify_roles(
            _records(),  # type: ignore[arg-type]
            transform_cutoff=1.0,
            max_travel=50.0,
            zncc_by_id=scores,
            identity_zncc_min=0.25)

    def test_unmeasurable_cells_are_identity_suspect(self):
        scores, _ = self._score(_masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)

        result = self._classify(scores)

        self.assertTrue(all(role == Role.IDENTITY_SUSPECT for role in result.roles))

    def test_unmeasurable_cells_are_not_counted_as_evaluated(self):
        """The old fabricated 0.0 was counted in n_zncc_eval as a real measurement."""
        scores, _ = self._score(_masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)

        result = self._classify(scores)

        self.assertEqual(result.n_zncc_eval, 0)
        self.assertEqual(result.n_zncc_fail, len(_POINTS))

    def test_reported_score_is_nan_not_zero(self):
        scores, _ = self._score(_masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)

        result = self._classify(scores)

        self.assertTrue(np.all(np.isnan(result.zncc)),
                        'a score that was never measured must not read as 0.0')

    def test_diagnostics_report_nan_for_a_missing_key(self):
        scores, _ = self._score(_masked_zncc_stack=_boom, _zncc_at_claimed_peak=_boom)

        # pass_diagnostics resolves absent keys this way.
        value = scores.get((0, 0), float('nan'))

        self.assertTrue(np.isnan(value))

    def test_healthy_candidates_still_lock(self):
        """Guard the premise: these records must be lockable when ZNCC works."""
        scores, _ = self._score()

        result = self._classify(scores)

        self.assertTrue(all(role == Role.LOCKABLE for role in result.roles))
        self.assertEqual(result.n_zncc_eval, len(_POINTS))


if __name__ == '__main__':
    unittest.main()
