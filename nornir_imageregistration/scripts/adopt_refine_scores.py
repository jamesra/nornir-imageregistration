'''Adopt an A/B run into the catalog living-best ledger.'''

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def __CreateArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Adopt a refine_runs.sqlite run into catalog best scores.')
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--fixtures-root', type=str, default=None)
    parser.add_argument('--fixture-id', type=int, default=None)
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--best-json', type=str, default=None,
                        help='Optional path to write best.json snapshot')
    return parser


def Execute(ExecArgs: list[str] | None = None) -> int:
    """Adopt one run; return process exit code."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
    args = __CreateArgParser().parse_args(ExecArgs)

    from nornir_imageregistration.refine_assessment.adopt import adopt_run
    from nornir_imageregistration.refine_assessment.catalog import (
        Catalog,
        RunsDb,
        catalog_path,
        runs_db_path,
    )

    fixtures_root = args.fixtures_root
    if fixtures_root is None:
        testinput = os.environ.get('TESTINPUTPATH')
        if not testinput:
            print('TESTINPUTPATH unset and --fixtures-root omitted', file=sys.stderr)
            return 2
        fixtures_root = str(Path(testinput) / 'refine_fixtures')

    catalog = Catalog.open(catalog_path(fixtures_root))
    runs = RunsDb.open(runs_db_path())
    try:
        best_path = Path(args.best_json) if args.best_json else None
        adopt_id = adopt_run(
            catalog,
            runs,
            args.run_id,
            fixture_id=args.fixture_id,
            note=args.note,
            best_json_path=best_path,
        )
        print(f'Adopted run {args.run_id} -> adopt_id={adopt_id}')
    finally:
        catalog.close()
        runs.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(Execute())
