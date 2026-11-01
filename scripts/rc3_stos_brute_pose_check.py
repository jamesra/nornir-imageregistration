#!/usr/bin/env python3
"""Measure StosBrute section-pair poses against reference Grid32 alignments (RC3 regression set).

Question answered: does a StosBrute variant (method, overlap, size, flip, masks, and the
capture-resolution ``WarpedImageScaleFactors``) land on the same pose as a trusted Grid32
alignment of the pair? Masked-NCC weight is not used: on low-content pairs it rewards a
precise wrong pose. The metric is the median distance, in ds32 pixels, between where the
brute pose and the reference Grid32 put a 7x7 grid of mapped-section points
(``y, x`` in 400..2800 step 400 at ds32; the brute is ds64, so points are halved, mapped,
and doubled). This is the metric from the RC3 2026-11-01 Build2 triage.

Inputs (``--data``; nothing is written there):

``img/<sec>_TEM_Blob.png``, ``img/<sec>_TEM_Mask.png``   ds64 Blob images and masks
``scale/<sec>.xml``            the section channel's ``VolumeData.xml`` (``<Scale>`` read
                               for ``--warped``)
``ref_apr2023|ref_feb2023|ref_dec2022/<pair>_ctrl-TEM_Leveled_map-TEM_Leveled.stos``
                               reference Grid32, first found wins
``new|mar2023|feb2023brute/<pair>_ctrl-TEM_Blob_map-TEM_Blob.stos``
                               stored brutes, reported with ``--stored``

``--pairs`` is a text file of ``<tag> <mapped>-<control>`` lines.

``--warped``:
  ``none``      ``WarpedImageScaleFactors=None``
  ``pipeline``  Control/Mapped UnitsPerPixel, what ``block._IsotropicWarpedImageScaleFactor``
                passes as of 2026-11-01
  ``physical``  Mapped/Control UnitsPerPixel, the direction ``stos_brute`` expects
                (see ``tests/test_SliceToSliceBrute.py`` ScaleMismatch tests)

Legacy ``Rigid2DTransform`` .stos files (pre-Nov-2023) are read with the angle negated.

Example::

    python -I scripts/rc3_stos_brute_pose_check.py --data C:/Temp/Nornir/rc3_brute/data \\
        --pairs pairs.txt --out results.tsv --variants pipeline --warped physical
"""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from typing import Any

import nornir_imageregistration
import numpy as np
from nornir_imageregistration import stos_brute
from nornir_imageregistration.files import stosfile
from nornir_imageregistration.settings import SliceToSliceMethod
from nornir_imageregistration.transforms import factory
from nornir_imageregistration.transforms.base import ITransform

PTS32: np.ndarray = np.array([[y, x] for y in range(400, 3200, 400) for x in range(400, 3200, 400)], float)
REFERENCE_DIRS: tuple[tuple[str, str], ...] = (('ref_apr2023', 'Apr2023'), ('ref_feb2023', 'Feb2023'),
                                               ('ref_dec2022', 'Dec2022'))
STORED_DIRS: tuple[str, ...] = ('new', 'mar2023', 'feb2023brute')
# Fixed random seed matching stos_brute_pairs.STOS_BRUTE_PAIR_SEED is not imported to keep
# this script independent of nornir_buildmanager; pass --seed to match a production run.
DEFAULT_SEED: int = 1852797550

_LOGPOLAR: dict[str, Any] = {'method': SliceToSliceMethod.LogPolar,
                               'MinOverlap': stos_brute.LOGPOLAR_PIPELINE_MIN_OVERLAP,
                               'LargestDimension': stos_brute.LOGPOLAR_PIPELINE_LARGEST_DIMENSION}
VARIANTS: dict[str, dict[str, Any]] = {
    'pipeline': dict(_LOGPOLAR),  # AlignSections default (-Method LogPolar)
    'lp_ov50': dict(_LOGPOLAR, MinOverlap=0.5),
    'lp_fullres': dict(_LOGPOLAR, LargestDimension=None),
    'lp_noflip': dict(_LOGPOLAR, TestFlip=False),
    'lp_nomask': dict(_LOGPOLAR, masks=False),
    'bf_818': {'method': SliceToSliceMethod.BruteForce, 'LargestDimension': 818},
    'bf_noscale_818': {'method': SliceToSliceMethod.BruteForce, 'LargestDimension': 818, 'search_scale': False},
    'bf': {'method': SliceToSliceMethod.BruteForce},
}


def _host(x: Any) -> np.ndarray:
    """Return *x* as a NumPy array (transforms follow the active backend)."""
    return nornir_imageregistration.EnsureNumpyArray(x)


def load_transform(path: str) -> ITransform:
    """Load a .stos transform; a legacy Rigid2D transform gets its angle negated."""
    parts = stosfile.StosFile.Load(path).Transform.split()
    if parts[0].startswith('Rigid2D'):
        return factory.ParseRigid2DTransform(parts, 1, negate_angle=True)
    return factory.LoadTransform(' '.join(parts), 1)


def find_reference(data: str, pair: str) -> tuple[ITransform | None, str]:
    """Return the first reference Grid32 found for *pair* and its label."""
    for sub, label in REFERENCE_DIRS:
        path = os.path.join(data, sub, f'{pair}_ctrl-TEM_Leveled_map-TEM_Leveled.stos')
        if os.path.exists(path):
            return load_transform(path), label
    return None, 'none'


def pose_distance(brute: ITransform, reference: ITransform) -> float:
    """Median distance (ds32 px) between a ds64 brute pose and a ds32 reference grid."""
    ref_points = _host(reference.Transform(PTS32))
    brute_points = _host(brute.Transform(PTS32 / 2.0)) * 2.0
    return float(np.median(np.linalg.norm(brute_points - ref_points, axis=1)))


def affine_summary(transform: ITransform) -> tuple[float, float, bool]:
    """Least-squares affine fit over the point grid: (angle degrees, isotropic scale, mirrored)."""
    src = PTS32 / 2.0
    dst = _host(transform.Transform(src))
    coeffs, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
    linear = coeffs[:2].T
    det = float(np.linalg.det(linear))
    angle = math.degrees(math.atan2(linear[1, 0] - linear[0, 1], linear[0, 0] + linear[1, 1]))
    return angle, math.sqrt(abs(det)), det < 0


def units_per_pixel(data: str, section: str) -> float:
    """X UnitsPerPixel from the section channel's ``<Scale>`` element."""
    with open(os.path.join(data, 'scale', f'{section}.xml'), encoding='utf-8') as f:
        text = f.read()
    block = re.search(r'<Scale[^>]*>(.*?)</Scale>', text, re.DOTALL)
    if block is None:
        raise ValueError(f'No <Scale> for section {section}')
    match = re.search(r'<X [^>]*UnitsPerPixel="([0-9.]+)"', block.group(1))
    if match is None:
        raise ValueError(f'No X UnitsPerPixel for section {section}')
    return float(match.group(1))


def warped_scale(data: str, mode: str, mapped: str, control: str) -> float | None:
    """``WarpedImageScaleFactors`` for *mode* (see module docstring)."""
    if mode == 'none':
        return None
    ratio = units_per_pixel(data, mapped) / units_per_pixel(data, control)
    return 1.0 / ratio if mode == 'pipeline' else ratio


def read_pairs(path: str) -> list[tuple[str, str]]:
    """Parse ``<tag> <mapped>-<control>`` lines."""
    with open(path, encoding='utf-8') as f:
        return [(parts[0], parts[1]) for parts in (line.split() for line in f) if len(parts) >= 2]


def report_stored(data: str, pairs: list[tuple[str, str]], out) -> None:
    """Write the pose distance of every stored brute that exists."""
    for tag, pair in pairs:
        reference, label = find_reference(data, pair)
        if reference is None:
            continue
        for sub in STORED_DIRS:
            path = os.path.join(data, sub, f'{pair}_ctrl-TEM_Blob_map-TEM_Blob.stos')
            if not os.path.exists(path):
                continue
            transform = load_transform(path)
            angle, scale, mirrored = affine_summary(transform)
            line = (f'stored:{sub}\t-\t-\t{tag}\t{pair}\t{label}\t{pose_distance(transform, reference):.0f}\t'
                    f'{angle:.1f}\t{scale:.3f}\t{int(mirrored)}\t-\t-')
            print(line, flush=True)
            out.write(line + '\n')


def run_variant(data: str, stage: str, name: str, mode: str, seed: int, backend: str,
                tag: str, pair: str) -> str:
    """Register one pair with one variant and return its result row."""
    settings = dict(VARIANTS[name])
    use_masks = settings.pop('masks', True)
    mapped, control = pair.split('-')
    img = os.path.join(data, 'img')
    target = os.path.join(img, f'{control}_TEM_Blob.png')
    source = os.path.join(img, f'{mapped}_TEM_Blob.png')
    target_mask = os.path.join(img, f'{control}_TEM_Mask.png') if use_masks else None
    source_mask = os.path.join(img, f'{mapped}_TEM_Mask.png') if use_masks else None
    scale = warped_scale(data, mode, mapped, control)

    nornir_imageregistration.seed_random_data(seed)
    started = time.perf_counter()
    record = stos_brute.SliceToSliceRigidRegistration(target, source, target_mask, source_mask,
                                                      AngleSearchRange=None,
                                                      WarpedImageScaleFactors=scale, **settings)
    elapsed = time.perf_counter() - started
    stos_path = os.path.join(stage, f'{name}_{mode}_{backend}_{pair}.stos')
    record.ToStos(target, source, target_mask, source_mask, PixelSpacing=64).Save(stos_path)

    transform = load_transform(stos_path)
    reference, label = find_reference(data, pair)
    distance = pose_distance(transform, reference) if reference is not None else float('nan')
    angle, fitted_scale, mirrored = affine_summary(transform)
    return (f'{name}+w:{mode}\t{backend}\t{seed}\t{tag}\t{pair}\t{label}\t{distance:.0f}\t{angle:.1f}\t'
            f'{fitted_scale:.3f}\t{int(mirrored)}\t{float(record.weight):.4g}\t{elapsed:.1f}')


def main() -> None:
    """Parse arguments and append result rows to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--data', required=True)
    parser.add_argument('--pairs', required=True, help='file of "<tag> <mapped>-<control>" lines')
    parser.add_argument('--out', required=True, help='TSV to append to; .stos go to <out dir>/stos')
    parser.add_argument('--variants', nargs='*', default=['pipeline'], choices=sorted(VARIANTS))
    parser.add_argument('--warped', choices=('none', 'pipeline', 'physical'), default='none')
    parser.add_argument('--only', default=None, help='comma-separated tags or pairs to run')
    parser.add_argument('--numpy', action='store_true', help='force the NumPy backend')
    parser.add_argument('--stored', action='store_true', help='also report the stored brutes')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.numpy:
        nornir_imageregistration.SetActiveComputationLib(nornir_imageregistration.ComputationLib.numpy)
    backend = 'numpy' if args.numpy else str(nornir_imageregistration.GetActiveComputationLib()).split('.')[-1]
    selected = set(args.only.split(',')) if args.only else None
    pairs = [(tag, pair) for tag, pair in read_pairs(args.pairs)
             if selected is None or tag in selected or pair in selected]
    stage = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'stos')
    os.makedirs(stage, exist_ok=True)

    with open(args.out, 'a', encoding='utf-8') as out:
        if args.stored:
            report_stored(args.data, pairs, out)
        for name in args.variants:
            for tag, pair in pairs:
                line = run_variant(args.data, stage, name, args.warped, args.seed, backend, tag, pair)
                print(line, flush=True)
                out.write(line + '\n')
                out.flush()


if __name__ == '__main__':
    main()
