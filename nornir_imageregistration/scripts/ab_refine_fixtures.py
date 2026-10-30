'''A/B refine fixtures: baseline vs NORNIR_REFINE_TRUSTED_MESH=1.'''

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def __CreateArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='A/B refine assessment fixtures and write a helped/broken report.')
    parser.add_argument('--fixtures-root', type=str, default=None,
                        help='Root of refine_fixtures (default $TESTINPUTPATH/refine_fixtures)')
    parser.add_argument('--fixture', type=str, action='append', default=[],
                        help='Relative fixture dir under fixtures-root (repeatable)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Run all fixtures with this confirmed tag')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Output work dir (default $TESTOUTPUTPATH/refine_ab)')
    parser.add_argument('--iterations', type=int, default=3)
    return parser


def Execute(ExecArgs: list[str] | None = None) -> int:
    """Run A/B across selected fixtures; return process exit code."""
    if ExecArgs is None:
        ExecArgs = sys.argv[1:]
    args = __CreateArgParser().parse_args(ExecArgs)

    from nornir_imageregistration.refine_assessment.ab_runner import run_ab_fixture, write_ab_report
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
    fixtures_root = Path(fixtures_root)

    work_dir = args.work_dir
    if work_dir is None:
        testoutput = os.environ.get('TESTOUTPUTPATH') or os.environ.get('TEST_OUTPUT_DIR')
        if not testoutput:
            print('TESTOUTPUTPATH unset and --work-dir omitted', file=sys.stderr)
            return 2
        work_dir = str(Path(testoutput) / 'refine_ab')
    work_dir = Path(work_dir)

    catalog = Catalog.open(catalog_path(fixtures_root))
    runs = RunsDb.open(runs_db_path())
    try:
        relative_dirs: list[str] = list(args.fixture)
        if args.tag:
            for row in catalog.list_fixtures(tag=args.tag, confirmed_only=True):
                relative_dirs.append(row['relative_dir'])
        if not relative_dirs:
            # Fall back to every registered fixture.
            relative_dirs = [row['relative_dir'] for row in catalog.list_fixtures()]
        if not relative_dirs:
            print('No fixtures selected', file=sys.stderr)
            return 1

        results = []
        for rel in relative_dirs:
            fixture_dir = fixtures_root / Path(rel)
            if not fixture_dir.is_dir():
                print(f'Skip missing fixture dir: {fixture_dir}', file=sys.stderr)
                continue
            print(f'A/B {rel} ...')
            results.append(run_ab_fixture(
                fixture_dir,
                catalog=catalog,
                runs=runs,
                work_dir=work_dir / rel.replace('/', '_'),
                relative_dir=rel,
                num_iterations=args.iterations,
            ))
            print(f'  verdict={results[-1].verdict}')

        csv_path, html_path = write_ab_report(results, work_dir)
        print(f'Wrote {csv_path}')
        print(f'Wrote {html_path}')
    finally:
        catalog.close()
        runs.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(Execute())
