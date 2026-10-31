"""
Noise-aware quantization ("crushing") of raw TEM tiles.

Shot-noise-limited TEM tiles carry many more intensity levels than their noise
lets anyone distinguish. Crushing stores ``floor((raw - offset) / step)`` with
``step = k * sigma`` counts, where sigma is the detector noise estimated from the
finest Haar wavelet subband. With ``k = 2.5`` a 16-bit tile keeps about 35-40
levels and fits in 8 bits, which cuts PNG size by ~86% against raw uint16.

The step is chosen once per section so every tile in a section shares one
intensity unit; per-tile steps would give neighbouring tiles different units and
break the section histogram, contrast and assembly. The parameters are stored in
each PNG (``tEXt`` chunk ``nornir_crush``) and on the volume's filter node, so the
quantized signal can be mapped back to approximate counts with :func:`uncrush`.

Host-only: tiles are read from and written to files with NumPy and Pillow.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image
from PIL.PngImagePlugin import PngInfo

CRUSH_VERSION = 1
DEFAULT_CRUSH_K = 2.5
SIGMA_METHOD = 'haar_hh_mad_center2048'
PNG_TEXT_KEY = 'nornir_crush'
PNG_COMPRESS_LEVEL = 6

# Donoho-Johnstone: MAD / 0.6745 estimates sigma for Gaussian noise.
_MAD_TO_SIGMA = 0.6745


@dataclasses.dataclass(frozen=True)
class CrushParams:
    """Section-wide crush parameters; ``counts ~= q * step + step / 2 + offset``."""
    k: float
    step: int
    sigma: float
    offset: int
    bits: int
    source_dtype: str
    sigma_method: str = SIGMA_METHOD
    version: int = CRUSH_VERSION

    @property
    def max_code(self) -> int:
        """Largest stored code; larger quotients are clipped to it."""
        return (1 << self.bits) - 1

    @property
    def storage_dtype(self) -> np.dtype:
        return np.dtype(np.uint8) if self.bits <= 8 else np.dtype(np.uint16)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CrushParams:
        values = json.loads(text)
        return cls(**{f.name: values[f.name] for f in dataclasses.fields(cls) if f.name in values})

    def to_attributes(self) -> dict[str, str]:
        """Attribute strings for a volume metadata node (``Crush*`` names)."""
        return {
            'CrushVersion': str(self.version),
            'CrushK': repr(float(self.k)),
            'CrushStep': str(self.step),
            'CrushSigma': repr(float(self.sigma)),
            'CrushOffset': str(self.offset),
            'CrushBits': str(self.bits),
            'CrushSourceDtype': self.source_dtype,
            'CrushSigmaMethod': self.sigma_method,
        }

    @classmethod
    def from_attributes(cls, attrib: Mapping[str, str]) -> CrushParams | None:
        """Inverse of :meth:`to_attributes`; None when the node was not crushed."""
        if 'CrushStep' not in attrib:
            return None

        return cls(k=float(attrib['CrushK']),
                   step=int(attrib['CrushStep']),
                   sigma=float(attrib['CrushSigma']),
                   offset=int(attrib['CrushOffset']),
                   bits=int(attrib['CrushBits']),
                   source_dtype=attrib['CrushSourceDtype'],
                   sigma_method=attrib.get('CrushSigmaMethod', SIGMA_METHOD),
                   version=int(attrib.get('CrushVersion', CRUSH_VERSION)))

    def same_quantization(self, other: CrushParams | None) -> bool:
        """True when *other* produces identical tiles (sigma itself may differ in its last digits)."""
        return other is not None and \
            (self.version, self.step, self.offset, self.bits) == (other.version, other.step, other.offset, other.bits)


def _center_crop(image: NDArray, crop: int) -> NDArray:
    """Even-aligned center crop of at most ``crop`` pixels per axis."""
    h = min(image.shape[0], crop) // 2 * 2
    w = min(image.shape[1], crop) // 2 * 2
    y0 = (image.shape[0] - h) // 4 * 2
    x0 = (image.shape[1] - w) // 4 * 2
    return image[y0:y0 + h, x0:x0 + w]


def estimate_noise_sigma(image: NDArray, crop: int = 2048) -> float:
    """
    Estimate the pixel noise sigma in counts with the Donoho-Johnstone estimator.

    The orthonormal Haar HH subband ``(a - b - c + d) / 2`` of 2x2 blocks keeps
    sigma for i.i.d. noise; its median absolute deviation / 0.6745 is robust to
    the structure that leaks into it. A center crop keeps the cost near 0.1 s per
    4k tile and stays within 3% of the full-tile estimate on RC2 tiles.
    """
    cropped = _center_crop(np.asarray(image), crop).astype(np.float64)
    hh = (cropped[0::2, 0::2] - cropped[0::2, 1::2] - cropped[1::2, 0::2] + cropped[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(hh - np.median(hh))) / _MAD_TO_SIGMA)


def section_sigma(sigmas: Sequence[float]) -> float:
    """Combine per-tile sigma estimates into one section value (median, robust to a bad tile)."""
    if len(sigmas) == 0:
        raise ValueError("No sigma estimates to combine")

    return float(np.median(np.asarray(sigmas, dtype=np.float64)))


def sample_indices(count: int, max_samples: int = 16) -> list[int]:
    """Indices of up to ``max_samples`` tiles spread evenly over ``count`` tiles."""
    if count <= max_samples:
        return list(range(count))

    return sorted({round(float(i)) for i in np.linspace(0, count - 1, max_samples)})


def make_crush_params(sigma: float, section_max: float, source_dtype: DTypeLike,
                      k: float = DEFAULT_CRUSH_K, section_min: float = 0) -> CrushParams:
    """
    Build section parameters from the section sigma and observed intensity range.

    ``offset`` is 0 for unsigned data and the (negative) section minimum for signed
    data, so signed MRC tiles keep their low counts. Storage is 8 bits whenever the
    largest quotient fits, otherwise the fewest bits that hold it in a uint16.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Noise sigma must be a positive finite number, got {sigma}")

    step = max(1, round(k * sigma))
    offset = min(0, math.floor(section_min)) if np.issubdtype(np.dtype(source_dtype), np.signedinteger) else 0
    largest_code = int(max(0.0, section_max - offset) // step)
    bits = 8 if largest_code <= 255 else max(9, largest_code.bit_length())
    if bits > 16:
        raise ValueError(f"Crushed codes need {bits} bits (step {step}); refusing to store them")

    return CrushParams(k=float(k), step=step, sigma=float(sigma), offset=offset, bits=bits,
                       source_dtype=np.dtype(source_dtype).name)


def crush(image: NDArray, params: CrushParams) -> NDArray:
    """Quantize raw counts to codes; values outside ``[offset, offset + max_code * step]`` are clipped."""
    shifted = np.asarray(image).astype(np.int64) - params.offset
    np.clip(shifted, 0, None, out=shifted)
    codes = shifted // params.step
    np.clip(codes, 0, params.max_code, out=codes)
    return codes.astype(params.storage_dtype)


def uncrush(codes: NDArray, params: CrushParams, dtype: Any = np.float32) -> NDArray:
    """Approximate raw counts from codes: the center of each code's bin."""
    return (np.asarray(codes).astype(np.float64) * params.step + params.step / 2.0 + params.offset).astype(dtype)


def save_crushed_png(path: str, codes: NDArray, params: CrushParams) -> None:
    """Write codes as a greyscale PNG carrying the parameters in a ``nornir_crush`` tEXt chunk."""
    info = PngInfo()
    info.add_text(PNG_TEXT_KEY, params.to_json())
    # Pillow infers 'L' from uint8 and 'I;16' from uint16.
    codes = np.ascontiguousarray(codes, dtype=params.storage_dtype)
    with Image.fromarray(codes) as im:
        im.save(path, format='PNG', compress_level=PNG_COMPRESS_LEVEL, pnginfo=info)


def read_crush_params(path: str) -> CrushParams | None:
    """Parameters stored in a crushed PNG, or None for any other image."""
    with Image.open(path) as im:
        text = getattr(im, 'text', {}).get(PNG_TEXT_KEY)

    return CrushParams.from_json(text) if text else None


def load_raw_tile(path: str) -> NDArray:
    """Load a source tile with its native integer dtype (no contrast, masking or rescaling)."""
    with Image.open(path) as im:
        return np.asarray(im)


def crush_tile_to_file(source_path: str, target_path: str, params: CrushParams, flip: bool = False) -> None:
    """
    Crush one source tile file into a target PNG.

    Top-level and picklable so process pools can run it. ``flip`` mirrors rows,
    matching ``Flip`` in ``ConvertImagesInDict`` (``xp.flip(image, 0)``).
    """
    image = load_raw_tile(source_path)
    if flip:
        image = np.flip(image, 0)

    os.makedirs(os.path.dirname(target_path) or '.', exist_ok=True)
    save_crushed_png(target_path, crush(image, params), params)
