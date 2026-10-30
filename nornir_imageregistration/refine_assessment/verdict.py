"""Score vectors and helped/unchanged/broken/still_flagged verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class MetricDirection(StrEnum):
    """How a metric should be compared to the adopted best."""

    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'
    BAND = 'band'


class Verdict(StrEnum):
    """A/B outcome relative to adopted bests and Manual gold."""

    HELPED = 'helped'
    UNCHANGED = 'unchanged'
    BROKEN = 'broken'
    STILL_FLAGGED = 'still_flagged'


# Default relative epsilon for maximize/minimize regressions (named-pair tunable).
DEFAULT_REL_EPS: float = 0.02


@dataclass(frozen=True)
class MetricSpec:
    """One scored quantity with comparison rules."""

    name: str
    direction: MetricDirection
    band_lo: float | None = None
    band_hi: float | None = None


# Canonical living-best metrics (pair ZNCC / lock_frac names match stos_quality).
CANONICAL_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec('pair_zncc', MetricDirection.MAXIMIZE),
    MetricSpec('unique_frac_min', MetricDirection.MAXIMIZE),
    MetricSpec('peak_ratio_median_certified', MetricDirection.MAXIMIZE),
    MetricSpec('peak_ratio_p10_certified', MetricDirection.MAXIMIZE),
    MetricSpec('zncc_prominence_median_certified', MetricDirection.MAXIMIZE),
    MetricSpec('lock_frac', MetricDirection.BAND, band_lo=0.29, band_hi=0.40),
    MetricSpec('certified_rmse', MetricDirection.MINIMIZE),
)


@dataclass
class ScoreVector:
    """Named metric values from one refine / A/B run."""

    values: dict[str, float] = field(default_factory=dict)
    quality_flag: bool = False
    unique_frac_series: list[float] = field(default_factory=list)

    def get(self, name: str) -> float | None:
        """Return metric *name* or None when absent."""
        if name not in self.values:
            return None
        return float(self.values[name])


def compare_to_best(
        scores: ScoreVector,
        best: Mapping[str, Mapping[str, float | str | None]],
        *,
        rel_eps: float = DEFAULT_REL_EPS,
        specs: tuple[MetricSpec, ...] = CANONICAL_METRICS,
) -> Verdict:
    """Compare *scores* to adopted *best* rows.

    *best* maps metric name → ``{value, direction, band_lo, band_hi}``.
    When there is no adopted best yet, returns ``STILL_FLAGGED`` if the quality
    flag is set, otherwise ``UNCHANGED``.
    """
    if not best:
        if scores.quality_flag:
            return Verdict.STILL_FLAGGED
        return Verdict.UNCHANGED

    any_helped = False
    any_broken = False
    for spec in specs:
        if spec.name not in best:
            continue
        current = scores.get(spec.name)
        if current is None:
            continue
        row = best[spec.name]
        try:
            adopted = float(row['value'])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        direction = str(row.get('direction', spec.direction))
        if direction == MetricDirection.MAXIMIZE:
            if current < adopted * (1.0 - rel_eps):
                any_broken = True
            elif current > adopted * (1.0 + rel_eps):
                any_helped = True
        elif direction == MetricDirection.MINIMIZE:
            if current > adopted * (1.0 + rel_eps):
                any_broken = True
            elif current < adopted * (1.0 - rel_eps):
                any_helped = True
        elif direction == MetricDirection.BAND:
            lo = row.get('band_lo', spec.band_lo)
            hi = row.get('band_hi', spec.band_hi)
            if lo is not None and hi is not None:
                if not (float(lo) <= current <= float(hi)):
                    any_broken = True

    series = scores.unique_frac_series
    if len(series) >= 2 and series[-1] < series[0] * (1.0 - rel_eps) and series[0] > 0.05:
        any_broken = True

    if any_broken:
        return Verdict.BROKEN
    if scores.quality_flag and not any_helped:
        return Verdict.STILL_FLAGGED
    if any_helped:
        return Verdict.HELPED
    return Verdict.UNCHANGED
