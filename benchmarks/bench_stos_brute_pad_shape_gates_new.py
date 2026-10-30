"""Pyperf worker entry: tuple shape gates (new)."""

from __future__ import annotations

import pyperf
from bench_stos_brute_pad_shape_checks import _make_arrays, bench_new

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_func("stos_brute_pad_shape_gates_new", bench_new, _make_arrays())
