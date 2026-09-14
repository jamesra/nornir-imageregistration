'''Confirm or suggest tags on a refine fixture in the SQLite catalog.'''

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def __CreateArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Tag a refine fixture in catalog.sqlite')
    parser.add_argument('--catalog', type=str, default=None,
                        help='Path to catalog.sqlite (default under TESTINPUTPATH)')
    parser.add_argument('--fixture-id', type=int, default=None)
    parser.add_argument('--relative-dir', type=str, default=None,
                        help='Fixture relative_dir key when id is unknown')
    parser.add_argument('--confirm', type=str, default='',
                        help='Comma-separated tags to mark confirmed')
    parser.add_argument('--suggest', type=str, default='',
                        help='Comma-separated tags to mark suggested')
    parser.add_argument('--list', action='store_true', help='List tags for the fixture')
    return parser


def Execute(ExecArgs: list[str] | None = None) -> int:
    """Run the tag CLI; return process exit code."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
    args = __CreateArgParser().parse_args(ExecArgs)

    from nornir_imageregistration.refine_assessment.catalog import Catalog, default_catalog_path
    from nornir_imageregistration.refine_assessment.tags import TagSource, TagStatus

    catalog_file = args.catalog
    if catalog_file is None:
        catalog_file = str(default_catalog_path())

    catalog = Catalog.open(catalog_file)
    try:
        fixture_id = args.fixture_id
        if fixture_id is None and args.relative_dir:
            fixture_id = catalog.fixture_id_for_dir(args.relative_dir)
        if fixture_id is None:
            print('Provide --fixture-id or --relative-dir', file=sys.stderr)
            return 2

        for slug in [t.strip() for t in args.confirm.split(',') if t.strip()]:
            catalog.tag_fixture(
                fixture_id, slug, status=TagStatus.CONFIRMED, source=TagSource.HUMAN)
            print(f'confirmed {slug} on fixture {fixture_id}')
        for slug in [t.strip() for t in args.suggest.split(',') if t.strip()]:
            catalog.tag_fixture(
                fixture_id, slug, status=TagStatus.SUGGESTED, source=TagSource.DIAGNOSTICS)
            print(f'suggested {slug} on fixture {fixture_id}')

        if args.list or (not args.confirm and not args.suggest):
            for tag in catalog.fixture_tags(fixture_id):
                print(f"{tag['slug']}\t{tag['status']}\t{tag['source']}")
    finally:
        catalog.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(Execute())
