'''Import a source .stos into TESTINPUTPATH/refine_fixtures.'''

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def __CreateArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Import a .stos (optional Manual + diagnostics) as a refine fixture.')
    parser.add_argument('--stos', type=str, default=None, help='Source .stos path')
    parser.add_argument('--manual', type=str, default=None, help='Optional Manual gold .stos')
    parser.add_argument('--diagnostics', type=str, default=None,
                        help='Optional refine_passNN_diagnostics.npz')
    parser.add_argument('--from-manual-dir', type=str, default=None,
                        help='Scan a StosGroup Manual/ folder and import each .stos')
    parser.add_argument('--out', type=str, default=None,
                        help='Fixtures root (default: $TESTINPUTPATH/refine_fixtures)')
    parser.add_argument('--volume', type=str, default='unknown')
    parser.add_argument('--group', type=str, default='Grid16')
    parser.add_argument('--pair', type=str, default=None)
    parser.add_argument('--channel', type=str, default='TEM')
    parser.add_argument('--bbox', type=float, nargs=4, default=None,
                        metavar=('Y0', 'X0', 'H', 'W'),
                        help='Crop bbox in source space')
    parser.add_argument('--auto-strain', action='store_true',
                        help='Pick crop from linear residual / unique diagnostics')
    parser.add_argument('--halo-hops', type=int, default=2)
    parser.add_argument('--pass-index', type=int, default=None)
    parser.add_argument('--tags', type=str, default='',
                        help='Comma-separated confirmed tags (e.g. tear,drying-front)')
    parser.add_argument('--candidate-source', type=str, default='named',
                        choices=('manual', 'diagnostics', 'named'))
    return parser


def Execute(ExecArgs: list[str] | None = None) -> int:
    """Run the importer CLI; return process exit code."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
    args = __CreateArgParser().parse_args(ExecArgs)

    from nornir_imageregistration.refine_assessment.catalog import Catalog, catalog_path
    from nornir_imageregistration.refine_assessment.importer import (
        import_refine_fixture,
        list_manual_stos,
    )

    out = args.out
    if out is None:
        testinput = os.environ.get('TESTINPUTPATH')
        if not testinput:
            print('TESTINPUTPATH is not set and --out was omitted', file=sys.stderr)
            return 2
        out = str(Path(testinput) / 'refine_fixtures')

    confirmed = [t.strip() for t in args.tags.split(',') if t.strip()]
    catalog = Catalog.open(catalog_path(out))
    try:
        paths: list[Path] = []
        if args.from_manual_dir:
            paths = list_manual_stos(args.from_manual_dir)
            if not paths:
                print(f'No .stos under {args.from_manual_dir}', file=sys.stderr)
                return 1
        elif args.stos:
            paths = [Path(args.stos)]
        else:
            print('Provide --stos or --from-manual-dir', file=sys.stderr)
            return 2

        for path in paths:
            result = import_refine_fixture(
                path,
                out,
                volume=args.volume,
                group_name=args.group,
                pair=args.pair,
                channel=args.channel,
                manual_path=args.manual,
                diagnostics_npz=args.diagnostics,
                bbox_yxhw=args.bbox,
                auto_strain=args.auto_strain,
                halo_hops=args.halo_hops,
                pass_index=args.pass_index,
                candidate_source=(
                    'manual' if args.from_manual_dir else args.candidate_source),
                confirmed_tags=confirmed,
                catalog=catalog,
            )
            print(
                f'Imported fixture_id={result.fixture_id} dir={result.fixture_dir} '
                f'suggested={list(result.suggested_tags)}')
    finally:
        catalog.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(Execute())
