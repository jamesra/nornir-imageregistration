"""Pairwise spatial distances with NumPy / CuPy dispatch.

``scipy.spatial.distance.cdist`` expects host arrays. When inputs are CuPy arrays
and GPU distance primitives are available (see ``HasCuVS``), use
``cupyx.scipy.spatial.distance.cdist`` (CuVS / pylibraft). Otherwise transfer
to the host and back.

There is no size gate on GPU ``cdist``. The full N×M matrix has no 2D-tree
shortcut, and the alternative for on-device arrays is a host round-trip that
is slower even at N=256 (CuVS ~0.40 ms vs H↔D SciPy ~1.30 ms). Host SciPy
wins at small N only when the data is already NumPy — that path is taken via
``get_array_module(XA)``.

Do not use this for k=1 nearest-point checks. Those go through
``nearest_neighbor.build_nearest_neighbor_index``, which keeps ``cKDTree``
below 4096 points because CuVS brute-force is O(N²) in 2D.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.spatial.distance

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp

try:
    import cupyx.scipy.spatial as _cupyx_spatial
except (ModuleNotFoundError, ImportError):
    _cupyx_spatial = None  # type: ignore[assignment]


def _to_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return np.asarray(a)
    getter = getattr(a, "get", None)
    if callable(getter):
        return np.asarray(getter())
    asnumpy = getattr(cp, "asnumpy", None)
    if callable(asnumpy):
        return np.asarray(asnumpy(a))
    return np.asarray(a)


def array_to_numpy_host(a: Any) -> np.ndarray:
    """Copy CuPy arrays to the host; return a NumPy ``ndarray``."""
    return _to_numpy(a)


def _cdist_same_dtype(XA: Any, XB: Any, xp: Any) -> tuple[Any, Any]:
    """Put *XB* on *xp* and match dtypes. CuVS pairwise_distance requires both.

    Mixed float32/float64 is downcast to float32: GPU control points are stored
    as float32, so promoting queries to float64 would invent precision the
    landmarks do not have.
    """
    XB = xp.asarray(XB)
    if XA.dtype == XB.dtype:
        return XA, XB
    dta = np.dtype(XA.dtype)
    dtb = np.dtype(XB.dtype)
    if np.issubdtype(dta, np.floating) and np.issubdtype(dtb, np.floating):
        dtype = dta if dta.itemsize <= dtb.itemsize else dtb
    else:
        dtype = np.promote_types(dta, dtb)
    return xp.asarray(XA, dtype=dtype), xp.asarray(XB, dtype=dtype)


def cdist(XA: Any, XB: Any, metric: str = "euclidean", **kwargs: Any) -> Any:
    """``cdist`` on the same device as ``XA`` (NumPy or CuPy).

    CuPy inputs use CuVS via CuPyX whenever ``HasCuVS()`` is true. Extra kwargs
    are forwarded to SciPy's ``cdist`` on the host path and to CuPyX where
    supported. Mixed float32/float64 inputs are downcast to float32 so CuVS
    does not raise ``Inputs must have the same dtypes``.
    """
    xp = cp.get_array_module(XA)
    if xp is np:
        return scipy.spatial.distance.cdist(XA, XB, metric, **kwargs)  # type: ignore[call-overload]

    from nornir_imageregistration.computational_lib import HasCuVS

    XA, XB = _cdist_same_dtype(XA, XB, xp)
    # CuVS pairwise_distance assumes C-contiguous (N, D) rows. Control-point
    # SourcePoints/TargetPoints are typically a (N, 2) view of a (N, 4) array
    # (float32 strides (16, 4)); CuVS then reads packed garbage distances and
    # TPS weight solves become NaN/Inf.
    XA = xp.ascontiguousarray(XA)
    XB = xp.ascontiguousarray(XB)
    if HasCuVS() and _cupyx_spatial is not None:
        return getattr(_cupyx_spatial, "distance").cdist(XA, XB, metric, **kwargs)  # type: ignore[attr-defined, call-overload]

    return cp.asarray(
        scipy.spatial.distance.cdist(_to_numpy(XA), _to_numpy(XB), metric, **kwargs)  # type: ignore[call-overload]
    )


__all__ = ["array_to_numpy_host", "cdist"]
