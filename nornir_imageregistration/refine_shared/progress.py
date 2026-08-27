"""MQTT dashboard progress helpers and pass-transform preview snapshots."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import nornir_imageregistration.settings
    import nornir_imageregistration.transforms
    from nornir_imageregistration.registration_control import ProgressCallback

_TRACK_PASSES = "stos_refine:passes"
_TRACK_LOCKED = "stos_refine:locked"
_TRACK_FILES = "stos_refine:files"


def _points_are_cupy(points: object) -> bool:
    """True when *points* live on the CuPy backend."""
    from nornir_imageregistration import cp

    return cp.get_array_module(points) is not np


def _cpu_mesh_from_points(points: object) -> "nornir_imageregistration.transforms.ITransform":
    """Build a CPU ``MeshWithRBFFallback`` from host-copied control points."""
    import nornir_imageregistration
    from nornir_imageregistration.transforms.meshwithrbffallback import MeshWithRBFFallback

    host_points = nornir_imageregistration.EnsureNumpyArray(points)
    return MeshWithRBFFallback(np.array(host_points, dtype=np.float64, copy=True))


def snapshot_transform_for_preview(
        transform: "nornir_imageregistration.transforms.ITransform | None",
) -> "nornir_imageregistration.transforms.ITransform | None":
    """Return an independent copy of *transform* for Qt/OpenGL preview.

    Refine continues to mutate the working transform on the worker thread.
    Preview callers must not share that live instance.

    Worker CuPy meshes cannot be deepcopied onto the GUI thread: the copy still
    owns device memory allocated on the worker, and ``paintGL`` then hits
    ``cudaErrorIllegalAddress``. Convert to a CPU mesh on this thread instead.
    """
    if transform is None:
        return None
    points = getattr(transform, "points", None)
    n_points = int(getattr(points, "shape", (0,))[0]) if points is not None else 0
    if n_points >= 3 and _points_are_cupy(points):
        return _cpu_mesh_from_points(points)
    try:
        snapshot = copy.deepcopy(transform)
        if snapshot is not transform:
            snap_points = getattr(snapshot, "points", None)
            if snap_points is not None and _points_are_cupy(snap_points):
                return _cpu_mesh_from_points(snap_points)
            return snapshot
    except Exception:
        pass
    if n_points >= 3:
        return _cpu_mesh_from_points(points)
    return None


def report_pass_transform(
        progress_callback: "ProgressCallback | None",
        transform: "nornir_imageregistration.transforms.ITransform | None",
        pass_index: int,
        num_iterations: int,
        *,
        label: str | None = None,
) -> None:
    """Snapshot *transform* and deliver it as a progress-callback preview."""
    if progress_callback is None or transform is None:
        return
    snapshot = snapshot_transform_for_preview(transform)
    if snapshot is None:
        return
    if label is None:
        label = f"Refine pass {pass_index}: updating view"
    from nornir_imageregistration.registration_control import report_progress

    report_progress(
        progress_callback,
        pass_index,
        num_iterations,
        label,
        preview=snapshot,
    )


def count_initial_grid_points(
    transform: "nornir_imageregistration.transforms.ITransform",
    settings: "nornir_imageregistration.settings.GridRefinement",
) -> int:
    """Return grid cell count after mask/bounds filtering (pass-1 denominator)."""
    import nornir_imageregistration.grid_subdivision

    grid_data = nornir_imageregistration.grid_subdivision.CenteredGridDivision(
        settings.source_image.shape,  # type: ignore[attr-defined]
        cell_size=settings.cell_size,
        grid_spacing=settings.grid_spacing,
        transform=transform,
    )
    grid_data.FilterOutofBoundsSourcePoints(settings.source_image.shape, allow_empty=False)
    source_mask = settings.source_mask
    if source_mask is None:
        source_mask = np.ones(settings.source_image.shape[:2], dtype=bool)
    grid_data.RemoveCellsUsingSourceImageMask(
        source_mask, settings.min_unmasked_area, allow_empty=False)
    if grid_data.num_points == 0:
        return 0

    grid_data.PopulateTargetPoints(transform)
    target_mask = settings.target_mask
    if target_mask is None:
        target_mask = np.ones(settings.target_image.shape[:2], dtype=bool)
    grid_data.FilterOutofBoundsTargetPoints(target_mask.shape, allow_empty=False)
    remaining = grid_data.RemoveCellsUsingTargetImageMask(
        target_mask, settings.min_unmasked_area, allow_empty=False)
    return int(remaining)


def publish_stos_refine_files_progress(
    *,
    current: int,
    total: int,
    label: str,
) -> None:
    """Publish outer-loop STOS-pair refine progress (depth 0)."""
    if total <= 0:
        return
    from nornir_shared import prettyoutput

    prettyoutput.publish_run_event(
        "iterate_progress",
        current=current,
        total=total,
        depth=0,
        track_id=_TRACK_FILES,
        label=label,
    )


class RefineGridProgressReporter:
    """Publish pass and locked-point progress for :func:`RefineTransform`."""

    def __init__(
        self,
        num_iterations: int,
        initial_grid_count: int,
        *,
        depth_base: int = 0,
    ) -> None:
        self._num_iterations = max(0, int(num_iterations))
        self._initial_grid_count = max(0, int(initial_grid_count))
        self._depth_base = int(depth_base)
        self._last_locked_count = 0
        self._last_active_count = 0
        self._last_pass_index = 0

    def on_pass_start(self, pass_index: int) -> None:
        """Publish refine-pass progress at the start of a pass."""
        if self._num_iterations <= 0:
            return
        from nornir_shared import prettyoutput

        self._last_pass_index = int(pass_index)
        prettyoutput.publish_run_event(
            "iterate_progress",
            current=pass_index,
            total=self._num_iterations,
            depth=self._depth_base,
            track_id=_TRACK_PASSES,
            label="Refine passes",
        )

    def on_pass_locked(self, pass_index: int, locked_count: int, active_count: int) -> None:
        """Publish locked-point progress after finalize stats for a pass."""
        if self._initial_grid_count <= 0:
            return
        from nornir_shared import prettyoutput

        locked = max(0, int(locked_count))
        active = max(0, int(active_count))
        self._last_locked_count = locked
        self._last_active_count = active
        self._last_pass_index = int(pass_index)
        prettyoutput.publish_run_event(
            "iterate_progress",
            current=locked,
            total=self._initial_grid_count,
            depth=self._depth_base + 1,
            track_id=_TRACK_LOCKED,
            label=(
                f"Locked {locked}/{self._initial_grid_count} "
                f"({locked}/{active} active)"
            ),
        )

    def on_complete(self) -> None:
        """Publish final pass and locked counts when refinement finishes."""
        if self._num_iterations > 0:
            from nornir_shared import prettyoutput

            prettyoutput.publish_run_event(
                "iterate_progress",
                current=self._num_iterations,
                total=self._num_iterations,
                depth=self._depth_base,
                track_id=_TRACK_PASSES,
                label="Refine passes",
            )
        if self._initial_grid_count > 0:
            self.on_pass_locked(
                self._last_pass_index,
                self._last_locked_count,
                self._last_active_count,
            )
