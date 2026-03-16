"""
GPU utilities for Hotspot. Provides CuPy availability checks and
sparse matrix construction helpers used by the GPU paths in
local_stats.py and local_stats_pairs.py.
"""

import numpy as np

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


def _build_sparse_weight_matrix(neighbors, weights, shape):
    """Build a CuPy sparse CSR matrix from neighbor/weight arrays.

    W[i, neighbors[i,k]] = weights[i,k]  for all i, k where weights[i,k] != 0.
    """
    N, K = neighbors.shape
    rows = np.repeat(np.arange(N, dtype=np.int32), K)
    cols = neighbors.ravel().astype(np.int32)
    vals = weights.ravel().astype(np.float64)

    mask = vals != 0
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    return cp_sparse.csr_matrix(
        (cp.asarray(vals), (cp.asarray(rows), cp.asarray(cols))),
        shape=shape,
    )


def _build_sparse_weight_sq_matrix(neighbors, weights, shape):
    """Build sparse matrix with squared weights: W_sq[i,j] = weights[i,k]^2."""
    N, K = neighbors.shape
    rows = np.repeat(np.arange(N, dtype=np.int32), K)
    cols = neighbors.ravel().astype(np.int32)
    vals = (weights.ravel().astype(np.float64)) ** 2

    mask = vals != 0
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    return cp_sparse.csr_matrix(
        (cp.asarray(vals), (cp.asarray(rows), cp.asarray(cols))),
        shape=shape,
    )
