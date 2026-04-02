"""Pairwise spatial distances with NumPy / CuPy dispatch.

``scipy.spatial.distance.cdist`` expects host arrays. When inputs are CuPy arrays and
GPU distance primitives are available (see ``HasCuVS``), use ``cupyx.scipy.spatial``.
Otherwise transfer to the host and back.
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


def cdist(XA: Any, XB: Any, metric: str = "euclidean", **kwargs: Any) -> Any:
    """``cdist`` on the same device as ``XA`` (NumPy or CuPy).

    Extra kwargs are forwarded to SciPy's ``cdist`` on the host path and to
    CuPyX where supported.
    """
    xp = cp.get_array_module(XA)
    if xp is np:
        return scipy.spatial.distance.cdist(XA, XB, metric, **kwargs)  # type: ignore[call-overload]

    from nornir_imageregistration.computational_lib import HasCuVS

    if HasCuVS() and _cupyx_spatial is not None:
        return getattr(_cupyx_spatial, "distance").cdist(XA, XB, metric, **kwargs)  # type: ignore[attr-defined, call-overload]

    return cp.asarray(
        scipy.spatial.distance.cdist(_to_numpy(XA), _to_numpy(XB), metric, **kwargs)  # type: ignore[call-overload]
    )


__all__ = ["array_to_numpy_host", "cdist"]
