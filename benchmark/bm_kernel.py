"""ctypes wrapper for the fused sparse Burer--Monteiro kernel."""

import ctypes
from pathlib import Path

import numpy as np
from scipy import sparse


class CompiledBmKernel:
    def __init__(self, cost, rank):
        cost = sparse.csr_matrix(cost, dtype=np.float64)
        cost.sort_indices()
        self._indptr = np.ascontiguousarray(cost.indptr, dtype=np.int32)
        self._indices = np.ascontiguousarray(cost.indices, dtype=np.int32)
        self._values = np.ascontiguousarray(cost.data, dtype=np.float64)
        self._rows = cost.shape[0]
        self._rank = int(rank)
        self._gradient = np.empty(self._rows * self._rank, dtype=np.float64)
        library = ctypes.CDLL(str(Path(__file__).with_name("libbm_kernel.so")))
        pointer_i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        pointer_f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        library.bm_kernel_create.restype = ctypes.c_void_p
        library.bm_kernel_create.argtypes = (
            ctypes.c_int, ctypes.c_int, pointer_i32, pointer_i32,
            pointer_f64, ctypes.c_int,
        )
        library.bm_kernel_evaluate.restype = ctypes.c_int
        library.bm_kernel_evaluate.argtypes = (
            ctypes.c_void_p, pointer_f64, pointer_f64,
            ctypes.POINTER(ctypes.c_double),
        )
        library.bm_kernel_destroy.argtypes = (ctypes.c_void_p,)
        self._library = library
        self._handle = library.bm_kernel_create(
            self._rows, self._rank, self._indptr, self._indices,
            self._values, self._values.size,
        )
        if not self._handle:
            raise RuntimeError("could not construct compiled BM kernel")

    def evaluate(self, flat_z):
        flat_z = np.ascontiguousarray(flat_z, dtype=np.float64)
        value = ctypes.c_double()
        status = self._library.bm_kernel_evaluate(
            self._handle, flat_z, self._gradient, ctypes.byref(value)
        )
        if status:
            raise RuntimeError(f"compiled BM evaluation failed with status {status}")
        return value.value, self._gradient

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.bm_kernel_destroy(handle)
            self._handle = None
