# Flip / Flop / mosaic coordinate contract

## Axes and origin

Nornir image and mosaic coordinates use **`(Y, X)`** ordering (row, column).
Image arrays are indexed `[y, x]`. Mosaic tile positions and transform
control points follow the same convention.

Origin is top-left of the image unless a specific importer documents otherwise.

## MosaicFile.Write Flip / Flop

[`MosaicFile.Write`](../nornir_imageregistration/files/mosaicfile.py) accepts:

| Flag | Effect on tile position `(X, Y)` before RigidTranslation |
|------|----------------------------------------------------------|
| `Flip=True` | Negate **Y** (`Y = -Y`) |
| `Flop=True` | Negate **X** (`X = -X`) |

These flags affect **mosaic placement coordinates only**. They do not flip
pixel data inside the tile files. Downsample / `ImageSize` arguments size the
per-tile transform footprint; they do not invert axes.

## Image convert Flip / Flop

When converting tiles (`ConvertImagesInDict*` / `_ConvertSingleImage`):

| Flag | Array axis |
|------|------------|
| Flip | axis **0** (rows / Y) |
| Flop | axis **1** (columns / X) |

GPU and CPU convert paths must agree (see `tests/test_convert_images_gpu_flip.py`).

## Transform Flip vs FlipWarped

| API | Space | Axis |
|-----|-------|------|
| `ControlPointBase.Flip` / GPU | Target **and** source | **X** about each space’s vertical midline |
| `*.FlipWarped(flip_center)` | Source only | **X** about `flip_center` (default mapped bbox center); restore **both** axes after negation |
| `Rigid.Flip` | Rigid model | Flip UD (Y) — different semantic from mosaic Write Flip |

Do not confuse mosaic Write `Flip` (negate Y positions) with transform
`FlipWarped` (mirror source X).

## Utah idoc import Y convention

SerialEM / Utah idoc import (`nornir_buildmanager.importers.idoc`) loads a
section flip list (`FlipList.txt`). When writing the section mosaic it passes:

```text
MosaicFile.Write(..., Flip=not Flip, ...)
```

Comment in code: “we flop instead of flip and reverse when writing the
coordinates.” A section listed in `FlipList.txt` therefore ends up with
**mosaic Write Flip=False** after the boolean invert (and the opposite when
absent). Treat this as the **Utah Y-invert contract** for TEM idoc imports;
PMG import uses `Flip=Flip` without that invert.

## Golden fixture

`tests/test_mosaicfile_flip_flop.py` locks MosaicFile.Write Flip/Flop sign
flips and documents the Utah `Flip=not Flip` interaction.
