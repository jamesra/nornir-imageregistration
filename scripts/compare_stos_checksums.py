#!/usr/bin/env python3
"""Compare .stos file checksums between two build output directories."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def stos_checksums(root: Path, pattern: str) -> dict[str, str]:
    """Return relative-path → sha256 hex digest for matching stos files."""
    digests: dict[str, str] = {}
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path, help='First volume/output directory')
    parser.add_argument('right', type=Path, help='Second volume/output directory')
    parser.add_argument('--pattern', default='TEM/Grid8/**/*.stos',
                        help='Glob pattern relative to each root')
    args = parser.parse_args()

    left = stos_checksums(args.left, args.pattern)
    right = stos_checksums(args.right, args.pattern)
    all_paths = sorted(set(left) | set(right))
    mismatches = 0
    for rel in all_paths:
        left_digest = left.get(rel)
        right_digest = right.get(rel)
        if left_digest == right_digest:
            status = 'MATCH'
        else:
            status = 'DIFFER'
            mismatches += 1
        print(f'{status}  {rel}')
        if left_digest != right_digest:
            print(f'         left={left_digest}')
            print(f'        right={right_digest}')
    print(f'Total: {len(all_paths)} files, {mismatches} mismatches')
    return 1 if mismatches else 0


if __name__ == '__main__':
    raise SystemExit(main())
