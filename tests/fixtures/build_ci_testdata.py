"""Build the ci-testdata fixture zip from a local TESTINPUTPATH corpus.

Run this script once on a developer machine that has the full nornir-testdata
corpus.  It selects and copies the files needed by B-class CI tests into a
staging tree, then zips them.

Usage::

    python tests/fixtures/build_ci_testdata.py [--out ci-testdata-v1.zip]

After running, upload the zip as a GitHub Release asset on the
nornir-imageregistration repository::

    gh release create ci-testdata-v1 ci-testdata-v1.zip \\
      --repo jamesra/nornir-imageregistration \\
      --title "CI test fixtures v1" \\
      --notes "B-class image fixtures for GitHub Actions CI (200 MB budget)"

Then ensure the ``key: ci-testdata-v1`` value in all affected workflow yml
files matches the release tag.  To update fixtures, bump to ``ci-testdata-v2``
(etc.) in both the release tag and workflow cache keys.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

BUDGET_MB = 200
BUDGET_BYTES = BUDGET_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Selection lists — add entries here when new B-class tests are wired in CI
# ---------------------------------------------------------------------------

# Individual files to copy from TESTINPUTPATH/Images/.
# Use missing_ok=True for files that are only needed by some test classes.
IMAGES: list[tuple[str, bool]] = [
    # shade correction / tile tests (test_tiles.py)
    ("400.png", False),
    ("400_Shaded.png", False),
    ("BrightfieldShading.png", False),
    ("CorrectionA_Tile.png", False),
    ("CorrectionA_Shading.png", False),
    ("CorrectionB_Tile.png", False),
    ("CorrectionB_Shading.png", False),
    # SliceToSliceBrute / registration (test_SliceToSliceBrute.py)
    ("0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png", False),
    ("0017_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8_Flipped.png", True),
    ("0017_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png", False),
    ("mini_TEM_Leveled_image__feabinary_Cel64_Mes8_sp4_Mes8.png", False),
    ("mini_TEM_Leveled_mask__feabinary_Cel64_Mes8_sp4_Mes8.png", False),
    # assemble Fixed/Moving (test_assemble.py, test_rigid_image_assembly.py)
    ("Fixed.png", False),
    ("Moving.png", False),
    # ImageAlign pair (test_ImageAlign.py)
    ("B029.png", False),
    ("B030.png", False),
    # AlignmentRecord IO (transforms/test_AlignmentRecord.py::TestIO)
    ("10x10.png", True),
    ("1000x100.png", True),
]

# Subdirectories of TESTINPUTPATH/Images/ to copy in full.
# Entries that don't exist on the local corpus are skipped with a warning.
IMAGE_SUBDIRS: list[str] = [
    "test_rigid",   # rigid_image_assembly test: Images/test_rigid/890.png etc.
]

# Individual files from TESTINPUTPATH/Transforms/.
TRANSFORMS: list[tuple[str, bool]] = [
    # assemble transform files (test_assemble.py)
    ("FixedMoving_Grid.stos", False),
    ("FixedMoving_Mesh.stos", False),
    # shared test_images.py + assemble composite check
    ("FixedMoving_Registered.png", False),
    # test_metrics.py
    ("0216-0215_grid_16.stos", False),
]

# Subdirectories of TESTINPUTPATH/Transforms/ to copy in full.
TRANSFORM_SUBDIRS: list[str] = [
    # test_addition.py (TestName = "TranslationTransformAddition")
    "TranslationTransformAddition",
    # test_assemble_tiles_basics.py (TestName = "PMG1" via TransformTestBase)
    "mosaics/PMG1",
]

_README_TEXT = """\
# CI test fixture pack

This zip is hosted as a GitHub Release asset on nornir-imageregistration.
It is downloaded by CI workflows (via actions/cache) and staged into
``TESTINPUTPATH/Images/`` and ``TESTINPUTPATH/Transforms/`` before B-class
tests run.

## Regenerating

1. Set ``TESTINPUTPATH`` to a full nornir-testdata corpus.
2. Run ``python tests/fixtures/build_ci_testdata.py``.
3. Upload the resulting zip to a new GitHub Release tag (e.g. ``ci-testdata-v2``).
4. Update ``key: ci-testdata-v2`` in all affected workflow yml files.

## Budget

Hard limit: {budget_mb} MB uncompressed.

## Intentionally excluded

- ``Images/Alignment/CaptureResolutionMismatch/`` — large; affected tests skip gracefully.
- Full DS1 image stacks.
- The full INPUT_NORNIR_DATA repro corpus.
- CuPy / GPU-specific test data (those tests skip when CuPy is unavailable).
""".format(budget_mb=BUDGET_MB)


def _copy_file(src: Path, dst: Path, missing_ok: bool = False) -> int:
    """Copy *src* to *dst*, creating parent dirs.

    Returns the number of bytes copied (0 if missing and missing_ok is True).
    """
    if not src.exists():
        verb = "SKIP (optional)" if missing_ok else "WARNING: missing"
        print(f"  {verb}: {src.name}")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size_kb = src.stat().st_size // 1024
    print(f"  {src.name}  ({size_kb} KB)")
    return src.stat().st_size


def _copy_tree(src: Path, dst: Path) -> int:
    """Copy an entire directory subtree from *src* into *dst*.

    Returns total uncompressed bytes copied.
    """
    if not src.exists():
        print(f"  WARNING: missing directory {src}")
        return 0
    total = 0
    for f in sorted(src.rglob("*")):
        if f.is_file():
            rel = f.relative_to(src)
            dst_f = dst / rel
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_f)
            total += f.stat().st_size
    size_mb = total / (1024 * 1024)
    print(f"  {src.name}/  ({size_mb:.1f} MB, {sum(1 for _ in src.rglob('*') if _.is_file())} files)")
    return total


def build(testinput: Path, out_zip: Path) -> None:
    """Stage fixtures from *testinput* and write *out_zip*."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        img_stage = stage / "Images"
        tfm_stage = stage / "Transforms"
        total_bytes = 0

        print("=== Staging Images/ ===")
        for name, missing_ok in IMAGES:
            total_bytes += _copy_file(
                testinput / "Images" / name,
                img_stage / name,
                missing_ok=missing_ok,
            )

        for subdir in IMAGE_SUBDIRS:
            total_bytes += _copy_tree(testinput / "Images" / subdir, img_stage / subdir)

        print("\n=== Staging Transforms/ ===")
        for name, missing_ok in TRANSFORMS:
            total_bytes += _copy_file(
                testinput / "Transforms" / name,
                tfm_stage / name,
                missing_ok=missing_ok,
            )

        for subdir in TRANSFORM_SUBDIRS:
            total_bytes += _copy_tree(
                testinput / "Transforms" / subdir,
                tfm_stage / subdir,
            )

        (stage / "README.md").write_text(_README_TEXT, encoding="utf-8")

        uncompressed_mb = total_bytes / (1024 * 1024)
        print(f"\nUncompressed staging tree: {uncompressed_mb:.1f} MB")

        if total_bytes > BUDGET_BYTES:
            print(f"\nERROR: {uncompressed_mb:.1f} MB exceeds {BUDGET_MB} MB budget — trim the selection lists above.")
            sys.exit(1)

        print("=== Writing zip ===")
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(stage))

        zip_mb = out_zip.stat().st_size / (1024 * 1024)
        print(f"Compressed zip:            {zip_mb:.1f} MB  →  {out_zip}")
        print()
        print("Upload to GitHub Release with:")
        print(f"  gh release create ci-testdata-v1 {out_zip} \\")
        print(f"    --repo jamesra/nornir-imageregistration \\")
        print(f"    --title 'CI test fixtures v1' \\")
        print(f"    --notes 'B-class image fixtures for GitHub Actions CI'")


def main() -> None:
    """Entry point for the build_ci_testdata script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testinput",
        default=os.environ.get("TESTINPUTPATH", ""),
        help="Path to full nornir-testdata corpus (defaults to $TESTINPUTPATH).",
    )
    parser.add_argument(
        "--out",
        default="ci-testdata-v1.zip",
        help="Output zip path (default: ci-testdata-v1.zip in CWD).",
    )
    args = parser.parse_args()

    if not args.testinput:
        print("ERROR: TESTINPUTPATH env var is not set and --testinput was not provided.")
        sys.exit(1)

    testinput = Path(args.testinput).expanduser().resolve()
    if not testinput.exists():
        print(f"ERROR: corpus path does not exist: {testinput}")
        sys.exit(1)

    out_zip = Path(args.out).resolve()
    print(f"Corpus:  {testinput}")
    print(f"Output:  {out_zip}")
    print(f"Budget:  {BUDGET_MB} MB uncompressed\n")

    build(testinput, out_zip)


if __name__ == "__main__":
    main()
