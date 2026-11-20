"""Masked band-pass + masked NCC (nornir_imageregistration.masked_correlation)."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.ndimage
from nornir_imageregistration import masked_correlation as mc
from nornir_imageregistration.phasecorrelation import find_offset

try:
    import cupy
    _HAVE_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001  # no CUDA on this host
    cupy = None
    _HAVE_CUPY = False


def _texture(shape: tuple[int, int], seed: int, sigma: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return scipy.ndimage.gaussian_filter(rng.standard_normal(shape), sigma)


def _brute_masked_ncc(f, mf, g, mg, min_overlap):
    """Direct loop: Pearson r over pixels valid in both, moving origin at (dy, dx) in fixed coords."""
    h1, w1 = f.shape
    h2, w2 = g.shape
    out = {}
    for dy in range(-(h2 - 1), h1):
        for dx in range(-(w2 - 1), w1):
            fy0, fy1 = max(0, dy), min(h1, dy + h2)
            fx0, fx1 = max(0, dx), min(w1, dx + w2)
            a = f[fy0:fy1, fx0:fx1]
            b = g[fy0 - dy:fy1 - dy, fx0 - dx:fx1 - dx]
            m = mf[fy0:fy1, fx0:fx1] & mg[fy0 - dy:fy1 - dy, fx0 - dx:fx1 - dx]
            if m.sum() < min_overlap:
                continue
            av, bv = a[m], b[m]
            if av.std() < 1e-9 or bv.std() < 1e-9:
                continue
            out[(dy, dx)] = float(np.corrcoef(av, bv)[0, 1])
    return out


class TestMaskedNccSurface(unittest.TestCase):

    def test_matches_a_direct_loop(self) -> None:
        rng = np.random.default_rng(3)
        f = rng.standard_normal((9, 11))
        g = rng.standard_normal((7, 8))
        mf = rng.random(f.shape) > 0.3
        mg = rng.random(g.shape) > 0.3
        ncc, _ = mc.masked_ncc_surface(f, mf, g, mg, min_overlap_fraction=0.2)
        min_overlap = max(0.2 * min(mf.sum(), mg.sum()), 2.0)
        expected = _brute_masked_ncc(f, mf, g, mg, min_overlap)

        centre = np.asarray(ncc.shape) // 2
        for (dy, dx), r in expected.items():
            # find_peak reports centre - index as the moving origin, so origin (dy, dx) sits at centre - (dy, dx).
            value = ncc[centre[0] - dy, centre[1] - dx]
            self.assertAlmostEqual(r, float(value), places=6, msg=f"shift {(dy, dx)}")
        self.assertEqual(len(expected), int(np.isfinite(ncc).sum()))

    def test_masked_pixel_values_do_not_matter(self) -> None:
        f = _texture((64, 64), 1)
        g = _texture((64, 64), 2)
        mf = np.ones(f.shape, bool)
        mf[:, :20] = False
        mg = np.ones(g.shape, bool)
        mg[40:, :] = False
        results = []
        for fill in (0.0, 1e3, np.nan):
            f2, g2 = f.copy(), g.copy()
            f2[~mf] = fill
            g2[~mg] = fill
            results.append(mc.masked_ncc_surface(f2, mf, g2, mg)[0])
        np.testing.assert_array_equal(np.isnan(results[0]), np.isnan(results[1]))
        np.testing.assert_allclose(results[0], results[1], atol=1e-9, equal_nan=True)
        np.testing.assert_allclose(results[0], results[2], atol=1e-9, equal_nan=True)

    def test_a_stack_gives_each_pair_its_own_surface(self) -> None:
        fs = np.stack([_texture((32, 32), s) for s in (1, 2, 3)])
        gs = np.stack([_texture((32, 32), s) for s in (4, 5, 6)])
        masks = np.ones(fs.shape, bool)
        masks[1, :10] = False
        stacked, _ = mc.masked_ncc_surface(fs, masks, gs, None)
        for i in range(3):
            single, _ = mc.masked_ncc_surface(fs[i], masks[i], gs[i], None)
            np.testing.assert_allclose(stacked[i], single, atol=1e-9, equal_nan=True)


class TestMaskedFindOffset(unittest.TestCase):

    def _pair(self, shift: tuple[int, int], seed: int = 11, size: int = 128):
        big = _texture((size + 64, size + 64), seed, sigma=1.5)
        fixed = big[32:32 + size, 32:32 + size]
        moving = big[32 + shift[0]:32 + shift[0] + size, 32 + shift[1]:32 + shift[1] + size]
        return fixed, moving

    def test_offset_convention_matches_find_offset(self) -> None:
        fixed, moving = self._pair((5, -9))
        old = find_offset(fixed.astype(np.float32), moving.astype(np.float32))
        new = mc.masked_find_offset(fixed, None, moving, None)
        np.testing.assert_allclose(new.record.peak, old.peak, atol=0.6)
        self.assertGreater(new.ncc, 0.95)

    def test_recovers_the_shift_with_much_of_the_window_masked(self) -> None:
        fixed, moving = self._pair((7, 4))
        expected = mc.masked_find_offset(fixed, None, moving, None).record.peak
        for masked_fraction in (0.3, 0.6):
            with self.subTest(masked_fraction=masked_fraction):
                cut = int(fixed.shape[1] * masked_fraction)
                mf = np.ones(fixed.shape, bool)
                mf[:, :cut] = False  # a tissue edge down the left side
                mg = np.ones(moving.shape, bool)
                mg[:, :cut] = False
                f = fixed.copy()
                f[~mf] = 0.0
                g = moving.copy()
                g[~mg] = 0.0
                result = mc.masked_find_offset(f, mf, g, mg)
                np.testing.assert_allclose(result.record.peak, expected, atol=0.6)
                self.assertGreater(result.ncc, 0.9)

    def test_bright_masked_areas_do_not_pull_the_peak(self) -> None:
        """A bright masked block at the same place in both crops (an edge at zero shift) is ignored."""
        fixed, moving = self._pair((0, 12))
        mf = np.ones(fixed.shape, bool)
        mf[:, :70] = False
        mg = mf.copy()
        f = fixed.copy()
        g = moving.copy()
        # The same masked "resin" block in both crops at the same place: an edge that sits at zero shift.
        f[~mf] = 5.0
        g[~mg] = 5.0
        masked = mc.masked_find_offset(f, mf, g, mg)
        truth = mc.masked_find_offset(fixed, None, moving, None).record.peak
        np.testing.assert_allclose(masked.record.peak, truth, atol=0.6)

    def test_is_deterministic(self) -> None:
        fixed, moving = self._pair((3, 3))
        mf = np.ones(fixed.shape, bool)
        mf[100:, :] = False
        a = mc.masked_find_offset(fixed, mf, moving, None)
        b = mc.masked_find_offset(fixed, mf, moving, None)
        np.testing.assert_array_equal(a.record.peak, b.record.peak)
        self.assertEqual(a.record.weight, b.record.weight)

    def test_fully_masked_gives_no_peak(self) -> None:
        fixed, moving = self._pair((0, 0))
        result = mc.masked_find_offset(fixed, np.zeros(fixed.shape, bool), moving, None)
        self.assertEqual(0.0, result.record.weight)
        self.assertTrue(np.isnan(result.ncc))


class TestMaskedNccSweep(unittest.TestCase):
    """One fixed image against many moving images with cached fixed spectra (StosBrute's angle sweep)."""

    def _bandpassed(self, image, mask):
        return mc.masked_bandpass(image, mask, *mc.default_bandpass_sigmas(image.shape))

    def test_matches_masked_find_offset_for_smaller_moving_images(self) -> None:
        big = _texture((260, 260), 21, sigma=1.5)
        fixed = big[30:230, 30:230]
        f, fv = self._bandpassed(fixed, None)
        fv = fv.copy()
        fv[:, :60] = False
        sweep = mc.MaskedNccSweep(f, fv, (190, 200), min_overlap_fraction=0.25)
        for shape, shift in (((190, 200), (6, -4)), ((150, 170), (-9, 13))):
            with self.subTest(shape=shape):
                y0, x0 = 30 + shift[0], 30 + shift[1]
                moving = big[y0:y0 + shape[0], x0:x0 + shape[1]]
                g, gv = self._bandpassed(moving, None)
                got = sweep.find_offset(g, gv)
                want = mc._offset_from_surface(*mc.masked_ncc_surface(f, fv, g, gv, 0.25),
                                               f.shape, g.shape, np)
                np.testing.assert_allclose(got.record.peak, want.record.peak, atol=0.05)
                self.assertAlmostEqual(got.ncc, want.ncc, places=3)
                self.assertEqual(got.overlap, want.overlap)

    def test_a_moving_image_larger_than_the_sweep_is_refused(self) -> None:
        f, fv = self._bandpassed(_texture((64, 64), 3), None)
        sweep = mc.MaskedNccSweep(f, fv, (40, 40))
        g, gv = self._bandpassed(_texture((41, 40), 4), None)
        with self.assertRaises(ValueError):
            sweep.find_offset(g, gv)


class TestSignificance(unittest.TestCase):

    def test_the_same_ncc_over_a_larger_overlap_ranks_higher(self) -> None:
        ncc = np.array([[0.6, 0.6, np.nan]])
        overlap = np.array([[100.0, 400.0, 400.0]])
        z = mc.significance_surface(ncc, overlap, np)
        self.assertGreater(z[0, 1], z[0, 0])
        self.assertTrue(np.isnan(z[0, 2]))
        surface, eligible = mc.surface_for_peak_finding(z, np)
        self.assertEqual(1.0, float(surface[0, 1]))
        np.testing.assert_array_equal([[True, True, False]], eligible)


class TestAttemptAlignPointMasked(unittest.TestCase):
    """The masked AttemptAlignPoint path recovers a known offset next to a tissue edge."""

    def test_known_offset_at_a_tissue_edge(self) -> None:
        from nornir_imageregistration.local_distortion_correction import (
            AttemptAlignPoint,
        )
        from nornir_imageregistration.transforms.rigid import RigidNoRotation

        big = _texture((640, 640), 21, sigma=1.5)
        target = big[64:576, 64:576].copy()
        source = big[64 + 6:576 + 6, 64 - 4:576 - 4].copy()
        target_mask = np.ones(target.shape, bool)
        target_mask[:, 300:] = False  # tissue ends right of x = 300
        source_mask = np.ones(source.shape, bool)
        source_mask[:, 304:] = False
        target[~target_mask] = 0.0
        source[~source_mask] = 0.0
        point = np.array([256.0, 290.0])  # window centred on the edge, about half tissue

        identity = RigidNoRotation(target_offset=(0.0, 0.0))
        unmasked = AttemptAlignPoint(identity, big[64:576, 64:576], big[70:582, 60:572], None, None, point,
                                     np.array([128, 128]), np.array([0.0]), estimate_angle=False,
                                     search_scale=False, target_mask=np.ones(target.shape, bool))
        masked = AttemptAlignPoint(identity, target, source, None, None, point, np.array([128, 128]),
                                   np.array([0.0]), estimate_angle=False, search_scale=False,
                                   target_mask=target_mask, source_mask=source_mask)
        assert unmasked is not None and masked is not None
        np.testing.assert_allclose(np.abs(masked.peak), [6.0, 4.0], atol=0.6)
        np.testing.assert_allclose(masked.peak, unmasked.peak, atol=0.6)
        self.assertGreater(masked.masked_ncc, 0.9)


class TestMaskedBandpass(unittest.TestCase):

    def test_mask_edge_does_not_leak(self) -> None:
        """A step in the masked area changes nothing in the valid area."""
        image = _texture((64, 64), 7)
        mask = np.ones(image.shape, bool)
        mask[:, 32:] = False
        a = image.copy()
        a[:, 32:] = 0.0
        b = image.copy()
        b[:, 32:] = 100.0
        fa, va = mc.masked_bandpass(a, mask, 1.0, 6.0)
        fb, vb = mc.masked_bandpass(b, mask, 1.0, 6.0)
        np.testing.assert_array_equal(va, vb)
        np.testing.assert_allclose(fa, fb, atol=1e-12)
        self.assertFalse(va[:, 32:].any())


@unittest.skipUnless(_HAVE_CUPY, "CuPy with a CUDA device is required")
class TestMaskedCorrelationCupyParity(unittest.TestCase):

    def test_cupy_matches_numpy(self) -> None:
        f = _texture((96, 96), 1)
        big = _texture((160, 160), 1)
        f = big[20:116, 20:116]
        g = big[26:122, 15:111]
        mf = np.ones(f.shape, bool)
        mf[:30] = False
        host = mc.masked_find_offset(f, mf, g, None)
        device = mc.masked_find_offset(cupy.asarray(f), cupy.asarray(mf), cupy.asarray(g), None)
        np.testing.assert_allclose(device.record.peak, host.record.peak, atol=1e-6)
        self.assertAlmostEqual(device.ncc, host.ncc, places=6)



class TestMaskedFindOffsetBatched(unittest.TestCase):
    """The batched form (refine grid cells) agrees with the serial one pair by pair."""

    def _stack(self):
        shifts = [(5, -9), (0, 0), (-12, 3), (7, 4)]
        cuts = [0, 20, 40, 60]  # masked columns down the left side: none, a little, much, most
        fixed, moving, fixed_mask, moving_mask = [], [], [], []
        for i, (shift, cut) in enumerate(zip(shifts, cuts)):
            f, g = TestMaskedFindOffset._pair(None, shift, seed=20 + i)  # type: ignore[arg-type]
            mf = np.ones(f.shape, bool)
            mf[:, :cut] = False
            mg = np.ones(g.shape, bool)
            mg[:, :cut] = False
            f, g = f.copy(), g.copy()
            f[~mf] = 0.0
            g[~mg] = 1e3  # masked values never take part
            fixed.append(f)
            moving.append(g)
            fixed_mask.append(mf)
            moving_mask.append(mg)
        return (np.stack(fixed), np.stack(fixed_mask), np.stack(moving), np.stack(moving_mask))

    def test_matches_the_serial_offsets(self) -> None:
        fixed, fixed_mask, moving, moving_mask = self._stack()
        batched = mc.masked_find_offset_batched(fixed, fixed_mask, moving, moving_mask)
        for i in range(fixed.shape[0]):
            with self.subTest(pair=i):
                serial = mc.masked_find_offset(fixed[i], fixed_mask[i], moving[i], moving_mask[i])
                np.testing.assert_allclose(batched.peaks[i], serial.record.peak, atol=0.25)
                self.assertAlmostEqual(float(batched.ncc[i]), serial.ncc, places=6)
                self.assertGreater(float(batched.weights[i]), 0.0)
                self.assertGreater(float(batched.peak_ratios[i]), 1.0)

    def test_a_fully_masked_pair_has_no_peak_and_leaves_the_others_alone(self) -> None:
        fixed, fixed_mask, moving, moving_mask = self._stack()
        whole = mc.masked_find_offset_batched(fixed, fixed_mask, moving, moving_mask)
        moving_mask = moving_mask.copy()
        moving_mask[1] = False
        result = mc.masked_find_offset_batched(fixed, fixed_mask, moving, moving_mask)
        self.assertEqual(float(result.weights[1]), 0.0)
        self.assertEqual(float(result.peak_ratios[1]), 0.0)
        self.assertTrue(np.isnan(float(result.ncc[1])))
        np.testing.assert_array_equal(result.peaks[1], (0.0, 0.0))
        for i in (0, 2, 3):
            np.testing.assert_allclose(result.peaks[i], whole.peaks[i], atol=1e-9)

    @unittest.skipUnless(_HAVE_CUPY, "needs CUDA")
    def test_cupy_matches_numpy(self) -> None:
        fixed, fixed_mask, moving, moving_mask = self._stack()
        host = mc.masked_find_offset_batched(fixed, fixed_mask, moving, moving_mask)
        device = mc.masked_find_offset_batched(cupy.asarray(fixed), cupy.asarray(fixed_mask),
                                               cupy.asarray(moving), cupy.asarray(moving_mask))
        np.testing.assert_allclose(cupy.asnumpy(device.peaks), host.peaks, atol=1e-6)
        np.testing.assert_allclose(cupy.asnumpy(device.ncc), host.ncc, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
