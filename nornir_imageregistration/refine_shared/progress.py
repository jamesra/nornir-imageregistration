"""MQTT dashboard progress helpers for STOS grid refinement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import nornir_imageregistration.settings
    import nornir_imageregistration.transforms

_TRACK_PASSES = "stos_refine:passes"
_TRACK_LOCKED = "stos_refine:locked"
_TRACK_FILES = "stos_refine:files"


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
