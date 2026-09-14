"""Controlled vocabulary for refine-fixture tags."""

from __future__ import annotations

from enum import StrEnum


class TagStatus(StrEnum):
    """Whether a fixture tag is metric-suggested or human-confirmed."""

    SUGGESTED = 'suggested'
    CONFIRMED = 'confirmed'


class TagSource(StrEnum):
    """How a fixture tag was attached."""

    HUMAN = 'human'
    DIAGNOSTICS = 'diagnostics'
    NAMED_DOC = 'named-doc'


# slug -> (description, suggestable)
SEED_TAGS: dict[str, tuple[str, bool]] = {
    'tear': ('Disc / tear front (cut-like discontinuity)', False),
    'fold': ('Fold or inverted mesh triangles', False),
    'drying-front': ('Band of unique cells with edges fanning out', False),
    'high-relative-distortion': ('Localized linear-residual / warp island', True),
    'coherent-residual': ('Uniform translation fringe across the FOV', True),
    'identity-freeze': ('Locks identity on the bad half of the field', True),
    'unique-collapse': ('Unique peak count falls across refine passes', True),
    'contrast-mismatch': ('Images disagree in intensity more than geometry', False),
    'white-stripe': ('Blank or overexposed stripe', False),
    'dirt': ('Dirt / LOW_CONTENT band', True),
    'healthy': ('Lock fraction in the healthy band; modest warp', True),
    'damage': ('Uniform soup; little unique tissue to register', True),
}


def ensure_seed_tags(conn) -> None:
    """Insert controlled vocabulary rows if missing (idempotent)."""
    for slug, (description, suggestable) in SEED_TAGS.items():
        conn.execute(
            '''
            INSERT OR IGNORE INTO tag (slug, description, suggestable)
            VALUES (?, ?, ?)
            ''',
            (slug, description, 1 if suggestable else 0),
        )
