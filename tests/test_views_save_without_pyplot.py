"""Layout and tile-overlap plots saved to a file must not touch pyplot, so they are safe on worker threads.

Layout relaxation plots its steps from pool threads (``views.plot_layout``), and the Mosaic pipeline writes
``Layout_Translated_*.svg`` through ``views.plot_tile_overlaps``. On a desktop nornir used to select QtAgg, and pyplot
with a GUI backend off the main thread crashes the process (RC2 1226 Mosaic died silently after writing the layout
SVG, 2026-11-05). Saving now uses a Figure of its own on the Agg canvas; output files are unchanged.
"""

from __future__ import annotations

import os
import tempfile
import threading
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nornir_imageregistration
from nornir_imageregistration import views


@pytest.fixture
def pyplot_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a view touched pyplot while saving to a file')

    for name in ('clf', 'gcf', 'gca', 'figure', 'savefig', 'close', 'show', 'scatter', 'plot', 'colorbar',
                 'title', 'xlim', 'ylim', 'axis', 'tight_layout'):
        monkeypatch.setattr(plt, name, forbidden)


def _tile(i: int, y: float, x: float):
    return types.SimpleNamespace(ID=i, FixedBoundingBox=nornir_imageregistration.Rectangle.CreateFromPointAndArea(
        (y, x), (100, 100)))


def _overlaps():
    tiles = [_tile(0, 0, 0), _tile(1, 0, 90), _tile(2, 90, 0)]

    def overlap(a, b):
        rect = nornir_imageregistration.Rectangle.Intersect(a.FixedBoundingBox, b.FixedBoundingBox)
        return types.SimpleNamespace(A=a, B=b, overlap=0.1, offset=np.array([1.0, 2.0]), overlapping_target_rect=rect)

    return [overlap(tiles[0], tiles[1]), overlap(tiles[0], tiles[2])]


def _layout():
    pos = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 90.0]), 2: np.array([90.0, 0.0])}
    return types.SimpleNamespace(
        GetPositions=lambda: np.array([pos[i] for i in range(3)]),
        WeightedNetTensionVectors=lambda: np.array([[0, 1.0, 2.0], [1, -1.0, 0.5], [2, 0.0, -2.0]]),
        MaxTensionMagnitude=(1, 2.5), MinTensionMagnitude=(0, 0.1),
        linked_nodes={(0, 1), (0, 2)}, GetPosition=lambda i: pos[i],
        PairTensionVector=lambda a, b: np.array([1.0, 1.0]))


def _run_on_worker(fn):
    errors: list[BaseException] = []

    def run():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
            errors.append(e)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=120)
    assert not worker.is_alive()
    assert not errors, errors


def test_tile_overlaps_svg_saves_on_a_worker_thread(pyplot_forbidden):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'Layout_Translated_Prune.svg')
        _run_on_worker(lambda: views.plot_tile_overlaps(_overlaps(), OutputFilename=path))
        assert os.path.getsize(path) > 0


def test_layout_saves_on_a_worker_thread(pyplot_forbidden):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, '0.svg')
        _run_on_worker(lambda: views.plot_layout(_layout(), shapes=['s', 'o', 's'], OutputFilename=path,
                                                 max_tension=2.5, title='step 0'))
        assert os.path.getsize(path) > 0
