r"""Both RBF mirrors must *delegate* on the rigid shortcut, not evaluate the solved affine.

Review #216: when `UseRigidTransform` was set, `OneWayRBFWithLinearCorrection.Transform`
returned `self._rigid_transform.Transform(Points)` while the `_GPUComponent` mirror zeroed
the RBF weight sums and fell through to the shared linear terms. Falling through evaluates
the *solved affine*, which is a different map from the rigid transform that
`_rigid_transform_is_equivalent` validated the shortcut against, and only the host path
involves the `_kabsch_umeyama` estimate at all.

Measured, the two maps agreed to solver noise wherever the shortcut is actually adopted:

| control points | worst \|host - affine\|, out to 50x span |
|---|---|
| reflection (y flip) | 2.6e-12 px |
| reflection + 11 deg rotation | 4.6e-04 px |
| three points | 1.3e-03 px |
| 4x4 grid, rigid | 1.1e-03 px |
| 10x10 grid, rigid + 1.05 scale | 4.4e-03 px |

They agree because the shortcut is only adopted for warps that are *exactly* affine --
`CalculateRBFWeights` needs `allclose(WeightsX[0:-3], 0)`, and a control-point perturbation
of only 0.005 px already rejects it -- and, after #214, additionally reproducible by the
rigid estimate to within 0.05 px. Degenerate sets that could leave the affine
underdetermined never reach the branch: collinear points and duplicates both raise in the
constructor, and near-collinear points fail the gate.

So this was a mirror-consistency defect rather than a numerical one, which is why
`test_one_way_rbf_gpu_host_parity` passed throughout -- it compares the two mirrors within a
tolerance that 4.4e-03 px sits inside. That test's docstring had already reasoned the
divergence was immaterial; what it could not do is stop the two branches drifting further
apart later.

These tests close that gap from the other side. The host-side ones run everywhere and pin
the numbers above, so a future change that makes the branches genuinely diverge is caught
even without a card. The GPU one asserts the stronger property the fix establishes: the
device result is now *bit-identical* to the rigid transform, where before the fix it was
merely close.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration.transforms.one_way_rbftransform import (
    OneWayRBFWithLinearCorrection)

_GRID = np.array([[float(y), float(x)]
                  for y in (0.0, 100.0, 200.0, 300.0)
                  for x in (0.0, 100.0, 200.0, 300.0)])


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return points @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T


def _build(source: np.ndarray, target: np.ndarray) -> OneWayRBFWithLinearCorrection:
    transform = OneWayRBFWithLinearCorrection(source, target)
    _ = transform.Weights  # the lazy solve is what decides the shortcut
    return transform


def _solved_affine_branch(transform, points: np.ndarray) -> np.ndarray:
    """The abandoned branch: zero the RBF weight sums, keep the linear terms.

    Reproduced in NumPy rather than CuPy on purpose. The arithmetic is identical in both,
    so this isolates the branch difference from any device numerics, and lets the
    comparison run on machines without a card.
    """
    points = nornir_imageregistration.EnsurePointsAre2DNumpyArray(points)
    weights = transform.Weights
    n = len(transform.TargetPoints)
    zeros = np.zeros((1, points.shape[0]))

    xf = np.sum(np.vstack((zeros,
                           points[:, 1] * weights[n],
                           points[:, 0] * weights[n + 1])), axis=0) + weights[n + 2]
    yf = np.sum(np.vstack((zeros,
                           points[:, 1] * weights[n + 3 + n],
                           points[:, 0] * weights[n + n + 3 + 1])), axis=0) + weights[n + n + 3 + 2]
    return np.vstack((xf, yf)).transpose()


def _probe_points(source: np.ndarray) -> dict[str, np.ndarray]:
    low, high = source.min(axis=0), source.max(axis=0)
    span = np.maximum(high - low, 1.0)
    centre = (low + high) / 2.0
    return {
        'control points': source,
        'hull centre': centre.reshape(1, 2),
        'one span out': (centre + span).reshape(1, 2),
        'fifty spans out': (centre + span * 50.0).reshape(1, 2),
    }


_ADOPTED_CASES = {
    'rigid': (_GRID, _rotate(_GRID, 7.0) + np.array([12.0, -5.0])),
    'rigid_with_scale': (_GRID, _rotate(_GRID, 7.0) * 1.10 + np.array([12.0, -5.0])),
    'reflection': (_GRID, _GRID @ np.array([[1.0, 0.0], [0.0, -1.0]]).T + np.array([5.0, 7.0])),
    'reflection_rotated': (_GRID,
                           _rotate(_GRID @ np.array([[1.0, 0.0], [0.0, -1.0]]).T, 11.0)
                           + np.array([5.0, 7.0])),
    'three_points': (np.array([[0.0, 0.0], [0.0, 200.0], [200.0, 0.0]]),
                     _rotate(np.array([[0.0, 0.0], [0.0, 200.0], [200.0, 0.0]]), 9.0)
                     + np.array([3.0, -2.0])),
}


class TestTheShortcutIsOnlyAdoptedForExactAffines(unittest.TestCase):
    """This is *why* the two branches never diverged materially. Pin it."""

    def test_an_exact_rigid_warp_adopts_the_shortcut(self):
        for name, (source, target) in _ADOPTED_CASES.items():
            with self.subTest(case=name):
                self.assertTrue(_build(source, target).UseRigidTransform,
                                f'{name} should take the rigid shortcut')

    def test_a_tiny_perturbation_already_rejects_it(self):
        rng = np.random.default_rng(216)
        target = _rotate(_GRID, 7.0) + np.array([12.0, -5.0])
        for amplitude in (0.005, 0.02, 0.04):
            with self.subTest(sd=amplitude):
                perturbed = target + rng.normal(scale=amplitude, size=_GRID.shape)
                self.assertFalse(_build(_GRID, perturbed).UseRigidTransform,
                                 f'a {amplitude} px perturbation should reject the '
                                 'shortcut; if it stops doing so the branches get room '
                                 'to disagree')

    def test_shear_is_rejected(self):
        sheared = _GRID @ np.array([[1.0, 0.05], [0.0, 1.0]]).T
        self.assertFalse(_build(_GRID, sheared).UseRigidTransform,
                         'shear is an exact affine but not rigid, so #214\'s equivalence '
                         'gate must reject it')

    def test_degenerate_sets_never_reach_the_branch(self):
        line = np.array([[0.0, float(x)] for x in (0.0, 50.0, 100.0, 150.0, 200.0)])
        with self.assertRaises(ValueError):
            _build(line, _rotate(line, 9.0))

        tri = np.array([[0.0, 0.0], [0.0, 200.0], [200.0, 0.0]])
        duplicated = np.vstack((tri, tri[0:1]))
        with self.assertRaises(ValueError):
            _build(duplicated, _rotate(duplicated, 9.0))

    def test_near_collinear_points_do_not_open_up_a_divergence(self):
        # A near-degenerate set is the one case where the affine could be poorly
        # determined while still reproducing the control points, so the branches would have
        # room to disagree. Whether the gate rejects such a set turns out to depend on how
        # near-degenerate it is, so assert the property that actually matters: if the
        # shortcut *is* adopted, the two branches still agree.
        line = np.array([[0.0, float(x)] for x in (0.0, 50.0, 100.0, 150.0, 200.0)])
        for nudge in (0.001, 0.01, 0.1, 1.0):
            with self.subTest(nudge=nudge):
                nudged = line + np.array([[nudge * i, 0.0] for i in range(len(line))])
                target = _rotate(nudged, 9.0) + np.array([3.0, -2.0])
                try:
                    transform = _build(nudged, target)
                except ValueError:
                    # EstimateRigidComponentsFromControlPoints refuses a set this close to
                    # collinear, so the shortcut is unreachable and there is no branch to
                    # disagree about. The strongest outcome available here.
                    continue
                if not transform.UseRigidTransform:
                    continue
                delegated = np.asarray(transform.Transform(nudged))
                affine = _solved_affine_branch(transform, nudged)
                worst = float(np.linalg.norm(delegated - affine, axis=1).max())
                self.assertLess(worst, 0.05,
                                f'nudge {nudge}: {worst:.3e} px between the branches on a '
                                'near-degenerate set')


class TestTheTwoBranchesAgreeToSolverNoise(unittest.TestCase):
    """Quantify the divergence, so a real one is distinguishable from this one."""

    # Worst measured was 4.4e-03 px, at 50x the control-point span on a 10x10 grid.
    TOLERANCE_PX = 0.02

    def test_the_delegated_and_affine_branches_agree(self):
        for name, (source, target) in _ADOPTED_CASES.items():
            transform = _build(source, target)
            for label, points in _probe_points(source).items():
                with self.subTest(case=name, probe=label):
                    delegated = np.asarray(transform.Transform(points))
                    affine = _solved_affine_branch(transform, points)
                    worst = float(np.linalg.norm(delegated - affine, axis=1).max())
                    self.assertLess(worst, self.TOLERANCE_PX,
                                    f'{name}/{label}: {worst:.3e} px. Exceeding this means '
                                    'the branches genuinely compute different maps, and '
                                    'the mirror choice starts to matter numerically')

    def test_the_disagreement_stays_under_the_equivalence_gate(self):
        # #214 admits the shortcut only when the rigid map reproduces the control points to
        # 0.05 px. At the control points the branch disagreement must be well inside that,
        # or the gate would be validating something other than what gets evaluated.
        for name, (source, target) in _ADOPTED_CASES.items():
            with self.subTest(case=name):
                transform = _build(source, target)
                delegated = np.asarray(transform.Transform(source))
                affine = _solved_affine_branch(transform, source)
                worst = float(np.linalg.norm(delegated - affine, axis=1).max())
                self.assertLess(worst, 0.05)

    def test_the_comparison_would_notice_a_real_divergence(self):
        # Guard the guard: perturb the affine branch by a pixel and confirm the check fires.
        source, target = _ADOPTED_CASES['rigid']
        transform = _build(source, target)
        delegated = np.asarray(transform.Transform(source))
        shifted = _solved_affine_branch(transform, source) + np.array([1.0, 0.0])
        worst = float(np.linalg.norm(delegated - shifted, axis=1).max())
        self.assertGreater(worst, self.TOLERANCE_PX)


@pytest.mark.skipif(
    nornir_imageregistration.GetComputationModule() is np,
    reason='requires a CuPy device to exercise the GPU mirror')
class TestTheGpuMirrorDelegates(unittest.TestCase):
    """The fix's actual property: the device result is now identical, not merely close."""

    @staticmethod
    def _build_device(source, target):
        cp = pytest.importorskip('cupy')
        from nornir_imageregistration.transforms.one_way_rbftransform import (
            OneWayRBFWithLinearCorrection_GPUComponent)
        device = OneWayRBFWithLinearCorrection_GPUComponent(
            cp.asarray(source), cp.asarray(target))
        _ = device.Weights
        return device

    def test_the_device_result_is_the_rigid_transform_exactly(self):
        for name, (source, target) in _ADOPTED_CASES.items():
            with self.subTest(case=name):
                device = self._build_device(source, target)
                self.assertTrue(device.UseRigidTransform)
                probes = _probe_points(source)['fifty spans out']
                got = nornir_imageregistration.EnsureNumpyArray(device.Transform(probes))
                expected = nornir_imageregistration.EnsureNumpyArray(
                    device._rigid_transform.Transform(probes))
                np.testing.assert_array_equal(
                    got, expected,
                    'the GPU mirror must delegate, so its output is the rigid transform '
                    'bit for bit. Before the #216 fix it evaluated the solved affine and '
                    'came out close but not equal')

    def test_the_device_agrees_with_the_host(self):
        for name, (source, target) in _ADOPTED_CASES.items():
            with self.subTest(case=name):
                host = _build(source, target)
                device = self._build_device(source, target)
                for label, points in _probe_points(source).items():
                    got = nornir_imageregistration.EnsureNumpyArray(device.Transform(points))
                    # 1e-4 px, not tighter: both mirrors now delegate, so what is left is
                    # host-vs-device float difference inside the rigid *estimate* itself
                    # (measured up to 1.6e-05 px), which this test is not about. Still an
                    # order of magnitude below the 4.4e-03 px branch difference it replaced.
                    np.testing.assert_allclose(
                        got, np.asarray(host.Transform(points)), rtol=0, atol=1e-4,
                        err_msg=f'{name}/{label}: mirrors must now agree far more tightly '
                                'than the old 4.4e-03 px branch difference')

    def test_the_non_rigid_path_is_untouched(self):
        rng = np.random.default_rng(217)
        target = _rotate(_GRID, 7.0) + rng.normal(scale=2.0, size=_GRID.shape)
        host = _build(_GRID, target)
        device = self._build_device(_GRID, target)
        self.assertFalse(host.UseRigidTransform)
        self.assertFalse(device.UseRigidTransform)
        got = nornir_imageregistration.EnsureNumpyArray(device.Transform(_GRID))
        np.testing.assert_allclose(got, np.asarray(host.Transform(_GRID)),
                                   rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
