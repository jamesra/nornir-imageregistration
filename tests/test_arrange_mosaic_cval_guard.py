"""The cval/dtype guard in __get_overlapping_image (#127).

The guard read:

    if cval is None: cval = 'random'
    if not np.issubdtype(dtype, np.floating) and np.isnan(cval): raise ValueError(...)

Two defects, both measured.

**np.isnan on the default.** ``cval`` defaults to None and is immediately turned into the
string ``'random'``. ``np.isnan('random')`` raises TypeError, so every non-floating dtype
crashed on entry instead of filling with noise. Eight of the 48 cval/dtype combinations probed
raised TypeError: ``{uint8, uint16, int32, bool}`` x ``{None, 'random'}``.

**It tested the wrong dtype.** The check ran *before* the block that resolves
``dtype = image.dtype``. The issue predicted the documented ``dtype=None`` path would raise,
but it does not: ``np.dtype(None)`` is ``float64``, so ``issubdtype`` is True and the isnan call
short-circuits away. That is worse than a crash — with ``dtype=None`` the guard passed on a
float64 it had invented, then went on to load a uint16 image, which is exactly the case it
existed to reject.

So it was simultaneously too eager to reject and unable to catch what it was for.

The guard now runs after dtype resolution and only evaluates isnan on a numeric cval. Reaching
it requires a non-floating dtype, and every current caller passes ``default_image_dtype()``
(float16) or ``np.float16``, so this is latent today and the tests drive the function directly.
"""

from __future__ import annotations

import unittest

import numpy as np

import nornir_imageregistration
import nornir_imageregistration.arrange_mosaic as arrange_mosaic

# Module-level private, so the leading double underscore is not mangled here.
_get_overlapping_image = getattr(arrange_mosaic, '_ArrangeMosaic__get_overlapping_image', None)
if _get_overlapping_image is None:
    _get_overlapping_image = getattr(arrange_mosaic, '__get_overlapping_image')

_RECT = nornir_imageregistration.Rectangle.CreateFromPointAndArea((4, 4), (8, 8))


def _image(dtype, shape=(32, 32)):
    """A small image with some structure, so noise fill is distinguishable from a constant."""
    values = np.arange(int(np.prod(shape))).reshape(shape) % 200
    if np.issubdtype(np.dtype(dtype), np.floating):
        return (values / 200.0).astype(dtype)
    return values.astype(dtype)


def _crop(dtype, cval, image_dtype=None, rect=_RECT, excess_scalar=1.0):
    """Call the function under test, requesting `dtype` on an image of `image_dtype`."""
    if image_dtype is None:
        image_dtype = np.float32 if dtype is None else dtype
    return _get_overlapping_image(_image(image_dtype), rect,
                                  excess_scalar=excess_scalar, cval=cval, dtype=dtype)


_INTEGER_DTYPES = [np.uint8, np.uint16, np.int32]
_FLOAT_DTYPES = [np.float16, np.float32, np.float64]


class TestTheRandomDefaultNoLongerCrashes(unittest.TestCase):
    """The eight combinations that raised TypeError."""

    def test_an_integer_dtype_accepts_the_cval_none_default(self):
        for dtype in _INTEGER_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                cropped = _crop(dtype, cval=None)
                self.assertEqual(np.dtype(dtype), cropped.dtype)

    def test_an_integer_dtype_accepts_cval_random_explicitly(self):
        for dtype in _INTEGER_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                cropped = _crop(dtype, cval='random')
                self.assertEqual(np.dtype(dtype), cropped.dtype)

    def test_a_bool_dtype_accepts_the_default(self):
        cropped = _crop(bool, cval=None)
        self.assertEqual(np.dtype(bool), cropped.dtype)

    def test_a_float_dtype_still_accepts_the_default(self):
        for dtype in _FLOAT_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                cropped = _crop(dtype, cval=None)
                self.assertEqual(np.dtype(dtype), cropped.dtype)

    def test_the_documented_dtype_none_default_works(self):
        cropped = _crop(None, cval=None, image_dtype=np.float32)
        self.assertEqual(np.dtype(np.float32), cropped.dtype)

    def test_a_float_dtype_reaches_the_random_fill(self):
        # A rect hanging off the edge is what makes cval matter at all.
        outside = nornir_imageregistration.Rectangle.CreateFromPointAndArea((-8, -8), (12, 12))
        for dtype in _FLOAT_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                cropped = _crop(dtype, cval='random', rect=outside)
                self.assertEqual(np.dtype(dtype), cropped.dtype)

    def test_an_integer_dtype_random_fill_succeeds(self):
        """#127 guard is clear; #248 made integer GenerateNoise work for off-edge crops."""
        outside = nornir_imageregistration.Rectangle.CreateFromPointAndArea((-8, -8), (12, 12))
        for dtype in _INTEGER_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                cropped = _crop(dtype, cval='random', rect=outside)
                self.assertEqual(np.dtype(dtype), cropped.dtype)
                self.assertEqual(cropped.shape, (12, 12))


class TestNanIntoAnIntegerIsStillRejected(unittest.TestCase):
    """The guard's actual purpose survives."""

    def test_nan_with_an_integer_dtype_raises(self):
        for dtype in _INTEGER_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                with self.assertRaises(ValueError):
                    _crop(dtype, cval=np.nan)

    def test_the_message_names_the_offending_dtype(self):
        with self.assertRaises(ValueError) as caught:
            _crop(np.uint16, cval=np.nan)
        self.assertIn('uint16', str(caught.exception))

    def test_it_is_a_valueerror_not_a_typeerror(self):
        # A TypeError here would mean isnan blew up again rather than the guard firing.
        for dtype in _INTEGER_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                try:
                    _crop(dtype, cval=np.nan)
                except ValueError:
                    pass
                except TypeError as error:
                    self.fail(f'isnan raised instead of the guard firing: {error}')

    def test_nan_with_a_float_dtype_is_allowed(self):
        for dtype in _FLOAT_DTYPES:
            with self.subTest(dtype=np.dtype(dtype).name):
                cropped = _crop(dtype, cval=np.nan)
                self.assertEqual(np.dtype(dtype), cropped.dtype)


class TestTheGuardNowSeesTheDtypeInUse(unittest.TestCase):
    """The half the guard could never catch before."""

    def test_nan_into_an_integer_image_under_dtype_none_is_caught(self):
        # dtype=None means the dtype comes from the image. Previously the guard tested
        # np.dtype(None) -- float64 -- passed, and let nan through to a uint16 image.
        for image_dtype in _INTEGER_DTYPES:
            with self.subTest(image_dtype=np.dtype(image_dtype).name):
                with self.assertRaises(ValueError):
                    _crop(None, cval=np.nan, image_dtype=image_dtype)

    def test_nan_into_a_float_image_under_dtype_none_is_allowed(self):
        for image_dtype in _FLOAT_DTYPES:
            with self.subTest(image_dtype=np.dtype(image_dtype).name):
                cropped = _crop(None, cval=np.nan, image_dtype=image_dtype)
                self.assertEqual(np.dtype(image_dtype), cropped.dtype)

    def test_the_old_guard_could_not_have_seen_this(self):
        # Premise: np.dtype(None) is float64, so the entry-time test was vacuous.
        self.assertEqual(np.dtype(np.float64), np.dtype(None))
        self.assertTrue(np.issubdtype(None, np.floating),
                        'the entry-time guard always short-circuited on the documented default')


class TestNumericCvalsAreUnaffected(unittest.TestCase):
    """Ordinary fills should behave exactly as before."""

    def test_zero_and_a_positive_value_pass_for_every_dtype(self):
        for dtype in _INTEGER_DTYPES + _FLOAT_DTYPES:
            for cval in (0, 0.0, 1):
                with self.subTest(dtype=np.dtype(dtype).name, cval=cval):
                    cropped = _crop(dtype, cval=cval)
                    self.assertEqual(np.dtype(dtype), cropped.dtype)

    def test_an_out_of_image_crop_fills_with_the_requested_value(self):
        outside = nornir_imageregistration.Rectangle.CreateFromPointAndArea((-16, -16), (8, 8))
        cropped = _crop(np.uint16, cval=7, rect=outside)
        self.assertTrue((cropped == 7).all(),
                        'a crop entirely outside the image should be all cval')

    def test_an_unsupported_string_is_still_rejected(self):
        # CropImage owns this rule; make sure the reordering did not bypass it.
        with self.assertRaises(ValueError):
            _crop(np.float32, cval='nonsense')


class TestTheMaskExtremaReturnShapeIsUnchanged(unittest.TestCase):
    """mask_extrema returns a tuple; the reordering must not disturb that."""

    def test_it_returns_the_image_and_the_mask(self):
        result = _get_overlapping_image(_image(np.float32), _RECT, excess_scalar=1.0,
                                        mask_extrema=True, cval='random', dtype=np.float32)
        self.assertEqual(2, len(result))
        cropped, mask = result
        self.assertEqual(cropped.shape, mask.shape)

    def test_it_returns_a_bare_image_without_mask_extrema(self):
        result = _get_overlapping_image(_image(np.float32), _RECT, excess_scalar=1.0,
                                        mask_extrema=False, cval='random', dtype=np.float32)
        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()
