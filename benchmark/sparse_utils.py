"""Sparse loading utilities for graph-structured QUBO instances."""

import numpy as np
from scipy import sparse


def load_problem_sparse(path: str):
    """Load the coordinate-form benchmark file directly into CSR format."""
    data = np.load(path)
    i = data["i"]
    j = data["j"]
    values = data["Jij"].astype(float, copy=False)
    n = int(max(np.max(i), np.max(j)) + 1)
    Q = sparse.coo_matrix((values, (i, j)), shape=(n, n)).tocsr()
    if (Q != Q.T).nnz:
        Q = Q + Q.T
    return Q.tocsr()
