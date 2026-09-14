"""STOS refine assessment corpus: fixtures, SQLite catalog, A/B, living bests."""

from nornir_imageregistration.refine_assessment.catalog import (
    Catalog,
    RunsDb,
    catalog_path,
    default_catalog_path,
    runs_db_path,
)
from nornir_imageregistration.refine_assessment.tags import (
    SEED_TAGS,
    TagStatus,
    ensure_seed_tags,
)
from nornir_imageregistration.refine_assessment.verdict import (
    MetricDirection,
    ScoreVector,
    Verdict,
    compare_to_best,
)

__all__ = [
    'Catalog',
    'RunsDb',
    'SEED_TAGS',
    'TagStatus',
    'MetricDirection',
    'ScoreVector',
    'Verdict',
    'catalog_path',
    'compare_to_best',
    'default_catalog_path',
    'ensure_seed_tags',
    'runs_db_path',
]
