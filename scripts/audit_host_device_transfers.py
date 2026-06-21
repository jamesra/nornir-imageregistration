#!/usr/bin/env python3
"""Summarize known host↔device transfer boundaries in imageregistration hot paths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HOTSPOT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\.get\s*\(", "Explicit CuPy → host (.get())"),
    (r"EnsureNumpyArray\s*\(", "CuPy → NumPy via EnsureNumpyArray"),
    (r"cp\.asarray\s*\(", "Host → CuPy upload (cp.asarray)"),
    (r"GetOverlapMask\s*\(", "Overlap mask (use GetOverlapMaskOnDevice when image is on GPU)"),
)

AUDIT_FILES: tuple[str, ...] = (
    "nornir_imageregistration/phasecorrelation.py",
    "nornir_imageregistration/stos_brute.py",
    "nornir_imageregistration/overlapmasking.py",
    "nornir_imageregistration/local_distortion_correction.py",
    "nornir_imageregistration/arrange_mosaic.py",
)


def scan_file(path: Path) -> list[str]:
    """Return formatted lines for each pattern match in *path*."""
    lines_out: list[str] = []
    text = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(text, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, label in HOTSPOT_PATTERNS:
            if re.search(pattern, line):
                lines_out.append(f"  {path.name}:{line_no} [{label}] {stripped[:100]}")
                break
    return lines_out


def main() -> int:
    """Print a static audit report for registration hot-path transfer sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="nornir-imageregistration package root",
    )
    args = parser.parse_args()

    print("Host↔device transfer audit (static scan)")
    print("=" * 60)
    print()
    print("Resolved in this pass:")
    print("  - GetOverlapMaskOnDevice: one GPU upload per overlap-mask geometry")
    print("  - find_peak: device percentile on CuPy; scalar export without EnsureNumpyArray")
    print("  - find_offset / ScoreOneAngle: use GetOverlapMaskOnDevice")
    print()
    print("Remaining intentional boundaries:")
    print("  - stos_brute log-polar: skimage CPU path (.get() once per call)")
    print("  - computational_lib: NumPy in multiprocessing child processes")
    print("  - AttemptAlignPoint: per-vertex ROI registration (future batching target)")
    print()
    print("Pattern scan:")
    total = 0
    for rel in AUDIT_FILES:
        path = args.root / rel
        if not path.is_file():
            continue
        matches = scan_file(path)
        if matches:
            print(f"\n{rel}:")
            for entry in matches:
                print(entry)
                total += 1
    print()
    print(f"Total flagged lines: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
