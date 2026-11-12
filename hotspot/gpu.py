"""
Backend utilities for Hotspot. The batched paths in local_stats.py and
local_stats_pairs.py run on numpy/scipy (CPU) or cupy (GPU).
"""

import numpy as np
import scipy.sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def is_gpu_available():
    """Check whether GPU acceleration is available (CuPy installed + CUDA device present)."""
    if not HAS_CUPY:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _require_gpu():
    """Raise an informative error if GPU is not available."""
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install with: pip install hotspotsc[gpu]  "
            "(or install cupy separately for your CUDA version, "
            "e.g. pip install cupy-cuda12x)"
        )
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as e:
        raise RuntimeError(
            "No CUDA-capable GPU device found. "
            "GPU acceleration requires an NVIDIA GPU with CUDA support."
        ) from e


def get_backend(use_gpu):
    """Return the array module (numpy or cupy) for the requested device."""
    if use_gpu:
        _require_gpu()
        return cp
    return np


def asnumpy(x):
    return cp.asnumpy(x) if HAS_CUPY else x


def _build_sparse_weight_matrix(neighbors, weights, shape, square=False, use_gpu=False):
    """Build a sparse CSR matrix (scipy, or CuPy if use_gpu) from neighbor/weight arrays.

    W[i, neighbors[i,k]] = weights[i,k]  for all i, k where weights[i,k] != 0.
    If square=True, uses weights^2 instead (for moment computations).
    """
    N, K = neighbors.shape
    rows = np.repeat(np.arange(N, dtype=np.int32), K)
    cols = neighbors.ravel().astype(np.int32)
    vals = weights.ravel().astype(np.float64)
    if square:
        vals = vals ** 2

    mask = vals != 0
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    W = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=shape)
    return cp_sparse.csr_matrix(W) if use_gpu else W
