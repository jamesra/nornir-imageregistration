#!/usr/bin/env python3
"""Legacy wrapper: forwards to ``microbench_grid_refine.py --mode mosaic``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

script = Path(__file__).resolve().with_name('microbench_grid_refine.py')
sys.argv = [str(script), '--mode', 'mosaic', *sys.argv[1:]]
runpy.run_path(str(script), run_name='__main__')
