"""Pyperf worker entry: np.array_equal shape gates (legacy)."""

from __future__ import annotations

import pyperf
from bench_stos_brute_pad_shape_checks import _make_arrays, bench_old

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_func("stos_brute_pad_shape_gates_old", bench_old, _make_arrays())
