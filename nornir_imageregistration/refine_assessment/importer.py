"""Import a source .stos into a local refine fixture under TESTINPUTPATH."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

import nornir_imageregistration
from nornir_imageregistration.files.stosfile import StosFile
from nornir_imageregistration.refine_assessment.catalog import Catalog, catalog_path
from nornir_imageregistration.refine_assessment.strain import (
    classify_struggle_vs_damage,
    pick_strain_crop_from_transform,
    suggest_tags_from_strain,
    unique_mask_from_diagnostics,
)
from nornir_imageregistration.refine_assessment.tags import TagSource, TagStatus
from nornir_imageregistration.transforms import IControlPoints, factory

_PAIR_RE = re.compile(r'(?P<a>\d+)-(?P<b>\d+)')


@dataclass(frozen=True)
class ImportResult:
    """Paths and catalog id produced by one fixture import."""

    fixture_dir: Path
    relative_dir: str
    fixture_id: int
    manifest_path: Path
    suggested_tags: tuple[str, ...]


def _parse_pair(name: str) -> str | None:
    match = _PAIR_RE.search(name)
    if match is None:
        return None
    return f"{match.group('a')}-{match.group('b')}"


def _crop_image(
        image: NDArray,
        y0: int,
        x0: int,
        height: int,
        width: int,
) -> NDArray:
    """Crop *image* to the given integer box (clipped to bounds)."""
    h, w = int(image.shape[0]), int(image.shape[1])
    y0c = max(0, min(y0, h - 1))
    x0c = max(0, min(x0, w - 1))
    y1c = max(y0c + 1, min(y0 + height, h))
    x1c = max(x0c + 1, min(x0 + width, w))
    return np.asarray(image[y0c:y1c, x0c:x1c])


def _create_shifted_transform(
        transform: IControlPoints,
        origin_yx: tuple[float, float],
) -> object:
    """Build a mesh/grid transform with points shifted by *origin_yx*."""
    oy, ox = float(origin_yx[0]), float(origin_yx[1])
    source = np.asarray(
        nornir_imageregistration.EnsureNumpyArray(transform.SourcePoints),
        dtype=np.float64).copy()
    target = np.asarray(
        nornir_imageregistration.EnsureNumpyArray(transform.TargetPoints),
        dtype=np.float64).copy()
    source[:, 0] -= oy
    source[:, 1] -= ox
    target[:, 0] -= oy
    target[:, 1] -= ox
    # Prefer MeshWithRBFFallback for arbitrary control sets.
    from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback
    points = np.hstack((target, source))
    return MeshWithRBFFallback(points)


def _write_cropped_stos(
        source_stos: StosFile,
        transform,
        out_path: Path,
        *,
        control_path: Path,
        mapped_path: Path,
        control_mask_path: Path | None,
        mapped_mask_path: Path | None,
        downsample: float | None,
        control_shape: tuple[int, int],
        mapped_shape: tuple[int, int],
) -> None:
    """Write a mini .stos referencing cropped sibling images."""
    out = StosFile()
    out.ControlImageFullPath = str(control_path)
    out.MappedImageFullPath = str(mapped_path)
    if control_mask_path is not None:
        out.ControlMaskFullPath = str(control_mask_path)
    if mapped_mask_path is not None:
        out.MappedMaskFullPath = str(mapped_mask_path)
    out.Downsample = downsample if downsample is not None else source_stos.Downsample
    out.ControlImageDim = [0.0, 0.0, float(control_shape[1]), float(control_shape[0])]
    out.MappedImageDim = [0.0, 0.0, float(mapped_shape[1]), float(mapped_shape[0])]
    out.Transform = transform
    out.Save(str(out_path), relative_paths=True)


def import_refine_fixture(
        stos_path: str | Path,
        out_root: str | Path,
        *,
        volume: str = 'unknown',
        group_name: str = 'Grid16',
        pair: str | None = None,
        channel: str = 'TEM',
        manual_path: str | Path | None = None,
        diagnostics_npz: str | Path | None = None,
        bbox_yxhw: Sequence[float] | None = None,
        auto_strain: bool = False,
        halo_hops: int = 2,
        grid_spacing: tuple[float, float] | None = None,
        pass_index: int | None = None,
        gold_kind: str = 'none',
        candidate_source: str = 'named',
        confirmed_tags: Sequence[str] | None = None,
        catalog: Catalog | None = None,
) -> ImportResult:
    """Copy / crop a source .stos into ``out_root/<volume>/<group>/<pair>/``.

    Registers the fixture in *catalog* (opened under *out_root* when omitted).
    """
    stos_path = Path(stos_path)
    out_root = Path(out_root)
    if not stos_path.is_file():
        raise FileNotFoundError(stos_path)

    pair_name = pair or _parse_pair(stos_path.name) or stos_path.stem
    relative_dir = f'{volume}/{group_name}/{pair_name}'.replace('\\', '/')
    fixture_dir = out_root / volume / group_name / pair_name
    fixture_dir.mkdir(parents=True, exist_ok=True)

    stos = StosFile.Load(str(stos_path))
    if stos is None or stos.Transform is None:
        raise ValueError(f'Could not load STOS: {stos_path}')
    stos.TryConvertRelativePathsToAbsolutePaths(str(stos_path.parent))

    transform = factory.LoadTransform(stos.Transform, 1)
    if transform is None:
        raise ValueError(f'Could not parse transform: {stos_path}')

    unique_mask = None
    if diagnostics_npz is not None:
        loaded = unique_mask_from_diagnostics(diagnostics_npz)
        if loaded is not None:
            unique_mask, _ = loaded
            shutil.copy2(diagnostics_npz, fixture_dir / Path(diagnostics_npz).name)

    crop = None
    if auto_strain and isinstance(transform, IControlPoints):
        crop = pick_strain_crop_from_transform(
            transform,
            grid_spacing=grid_spacing,
            halo_hops=halo_hops,
            unique_mask=unique_mask,
        )
        if crop is not None:
            bbox_yxhw = crop.bbox_yxhw

    control = nornir_imageregistration.ImageParamToImageArray(
        stos.ControlImageFullPath, dtype=nornir_imageregistration.default_image_dtype())
    mapped = nornir_imageregistration.ImageParamToImageArray(
        stos.MappedImageFullPath, dtype=nornir_imageregistration.default_image_dtype())
    control = nornir_imageregistration.EnsureNumpyArray(control)
    mapped = nornir_imageregistration.EnsureNumpyArray(mapped)

    origin = (0.0, 0.0)
    if bbox_yxhw is not None:
        y0, x0, h, w = (float(v) for v in bbox_yxhw)
        origin = (y0, x0)
        iy0, ix0 = int(np.floor(y0)), int(np.floor(x0))
        ih, iw = max(1, int(np.ceil(h))), max(1, int(np.ceil(w)))
        control = _crop_image(control, iy0, ix0, ih, iw)
        mapped = _crop_image(mapped, iy0, ix0, ih, iw)
        if isinstance(transform, IControlPoints):
            transform = _create_shifted_transform(transform, origin)

    control_path = fixture_dir / 'control.png'
    mapped_path = fixture_dir / 'mapped.png'
    nornir_imageregistration.SaveImage(str(control_path), control)
    nornir_imageregistration.SaveImage(str(mapped_path), mapped)

    control_mask_path = None
    mapped_mask_path = None
    if stos.ControlMaskFullPath and os.path.isfile(stos.ControlMaskFullPath):
        mask = nornir_imageregistration.EnsureNumpyArray(
            nornir_imageregistration.ImageParamToImageArray(stos.ControlMaskFullPath))
        if bbox_yxhw is not None:
            y0, x0, h, w = (float(v) for v in bbox_yxhw)
            mask = _crop_image(mask, int(y0), int(x0), int(np.ceil(h)), int(np.ceil(w)))
        control_mask_path = fixture_dir / 'control_mask.png'
        nornir_imageregistration.SaveImage(str(control_mask_path), mask)
    if stos.MappedMaskFullPath and os.path.isfile(stos.MappedMaskFullPath):
        mask = nornir_imageregistration.EnsureNumpyArray(
            nornir_imageregistration.ImageParamToImageArray(stos.MappedMaskFullPath))
        if bbox_yxhw is not None:
            y0, x0, h, w = (float(v) for v in bbox_yxhw)
            mask = _crop_image(mask, int(y0), int(x0), int(np.ceil(h)), int(np.ceil(w)))
        mapped_mask_path = fixture_dir / 'mapped_mask.png'
        nornir_imageregistration.SaveImage(str(mapped_mask_path), mask)

    input_stos_path = fixture_dir / 'input.stos'
    _write_cropped_stos(
        stos, transform, input_stos_path,
        control_path=control_path,
        mapped_path=mapped_path,
        control_mask_path=control_mask_path,
        mapped_mask_path=mapped_mask_path,
        downsample=float(stos.Downsample) if stos.Downsample is not None else None,
        control_shape=(int(control.shape[0]), int(control.shape[1])),
        mapped_shape=(int(mapped.shape[0]), int(mapped.shape[1])),
    )

    if manual_path is not None and Path(manual_path).is_file():
        gold_kind = gold_kind if gold_kind != 'none' else 'manual'
        manual = StosFile.Load(str(manual_path))
        gold_transform = transform
        if manual is not None and manual.Transform is not None:
            manual.TryConvertRelativePathsToAbsolutePaths(str(Path(manual_path).parent))
            loaded_gold = factory.LoadTransform(manual.Transform, 1)
            if loaded_gold is not None:
                gold_transform = loaded_gold
                if bbox_yxhw is not None and isinstance(gold_transform, IControlPoints):
                    gold_transform = _create_shifted_transform(gold_transform, origin)
        _write_cropped_stos(
            stos, gold_transform,
            fixture_dir / 'gold.stos',
            control_path=control_path,
            mapped_path=mapped_path,
            control_mask_path=control_mask_path,
            mapped_mask_path=mapped_mask_path,
            downsample=float(stos.Downsample) if stos.Downsample is not None else None,
            control_shape=(int(control.shape[0]), int(control.shape[1])),
            mapped_shape=(int(mapped.shape[0]), int(mapped.shape[1])),
        )

    suggested: list[str] = []
    struggle = 'unknown'
    if crop is not None:
        suggested = suggest_tags_from_strain(crop)
        struggle = 'struggle' if crop.localized else (
            'damage' if crop.unique_frac_in_crop < 0.02 else 'unknown')

    confirmed = list(confirmed_tags or [])
    manifest = {
        'volume': volume,
        'group': group_name,
        'pair': pair_name,
        'channel': channel,
        'downsample': float(stos.Downsample) if stos.Downsample is not None else None,
        'source_stos': str(stos_path),
        'source_stos_checksum': str(stos.Checksum or ''),
        'bbox_yxhw': list(bbox_yxhw) if bbox_yxhw is not None else None,
        'halo_hops': int(halo_hops),
        'pass_index': pass_index,
        'gold_kind': gold_kind,
        'candidate_source': candidate_source,
        'struggle_vs_damage': struggle,
        'confirmed_tags': confirmed,
        'suggested_tags': suggested,
        'certified_ids': list(crop.certified_ids) if crop is not None else [],
    }
    manifest_path = fixture_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    own_catalog = catalog is None
    if catalog is None:
        catalog = Catalog.open(catalog_path(out_root))
    try:
        fixture_id = catalog.upsert_fixture(
            relative_dir,
            volume=volume,
            group_name=group_name,
            downsample=float(stos.Downsample) if stos.Downsample is not None else None,
            pair=pair_name,
            channel=channel,
            gold_kind=gold_kind,
            candidate_source=candidate_source,
            struggle_vs_damage=struggle,
            source_stos_checksum=str(stos.Checksum or ''),
            bbox=bbox_yxhw,
            halo_hops=halo_hops,
            pass_index=pass_index,
        )
        if crop is not None:
            catalog.set_certified_cells(
                fixture_id,
                [(r, c, None, None) for r, c in crop.certified_ids],
            )
        for slug in suggested:
            catalog.tag_fixture(
                fixture_id, slug, status=TagStatus.SUGGESTED, source=TagSource.DIAGNOSTICS)
        for slug in confirmed:
            catalog.tag_fixture(
                fixture_id, slug, status=TagStatus.CONFIRMED, source=TagSource.HUMAN)
    finally:
        if own_catalog:
            catalog.close()

    return ImportResult(
        fixture_dir=fixture_dir,
        relative_dir=relative_dir,
        fixture_id=fixture_id,
        manifest_path=manifest_path,
        suggested_tags=tuple(suggested),
    )


def list_manual_stos(manual_dir: str | Path) -> list[Path]:
    """Return ``*.stos`` paths under a StosGroup ``Manual/`` folder."""
    manual_dir = Path(manual_dir)
    if not manual_dir.is_dir():
        return []
    return sorted(manual_dir.glob('*.stos'))
