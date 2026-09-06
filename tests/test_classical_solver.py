import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from simple_solver import (  # noqa: E402
    _qubo_sdp_cost_matrix,
    successive_burer_monteiro_minimize,
)
from sparse_utils import load_problem_sparse  # noqa: E402


class SparseLoaderTest(unittest.TestCase):
    def test_triangular_input_is_symmetrized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problem.npz"
            np.savez(
                path,
                i=np.array([0, 0, 1]),
                j=np.array([0, 1, 1]),
                Jij=np.array([-1.0, 2.0, -3.0]),
            )
            Q = load_problem_sparse(path)
        np.testing.assert_array_equal(
            Q.toarray(), np.array([[-2.0, 2.0], [2.0, -6.0]])
        )


class SuccessiveBmTest(unittest.TestCase):
    def setUp(self):
        self.Q = sparse.csr_matrix(
            np.array(
                [
                    [-3.0, 1.0, 0.0, 0.5],
                    [1.0, -2.0, -0.5, 0.0],
                    [0.0, -0.5, -2.0, 1.0],
                    [0.5, 0.0, 1.0, -3.0],
                ]
            )
        )

    def test_sparse_sdp_cost_matches_dense_formula(self):
        cost = _qubo_sdp_cost_matrix(self.Q)
        self.assertTrue(sparse.issparse(cost))
        ones = np.ones(self.Q.shape[0])
        dense_Q = self.Q.toarray()
        expected = np.block(
            [
                [0.25 * dense_Q, 0.25 * (dense_Q @ ones)[:, None]],
                [
                    0.25 * (ones @ dense_Q)[None, :],
                    np.array([[0.25 * ones @ dense_Q @ ones]]),
                ],
            ]
        )
        np.testing.assert_allclose(cost.toarray(), expected)

    def test_solver_returns_a_binary_nontrivial_solution(self):
        x, info = successive_burer_monteiro_minimize(
            self.Q,
            rank=3,
            time_limit=0.25,
            seed=7,
            relaxation_restarts=1,
            max_stages=3,
            max_iterations=8,
            return_info=True,
        )
        self.assertEqual(x.shape, (self.Q.shape[0],))
        self.assertTrue(np.all((x == 0) | (x == 1)))
        self.assertLessEqual(float(x @ (self.Q @ x)), 0.0)
        self.assertIn("objective_evaluations", info)


if __name__ == "__main__":
    unittest.main()
