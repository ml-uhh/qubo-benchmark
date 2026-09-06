"""Run the sparse successive Burer-Monteiro QUBO heuristic on one instance."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from simple_solver import successive_burer_monteiro_minimize
from sparse_utils import load_problem_sparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="coordinate-form .npz file")
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=35)
    parser.add_argument("--relaxation-restarts", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")

    Q = load_problem_sparse(args.instance)
    started = time.perf_counter()
    x, info = successive_burer_monteiro_minimize(
        Q,
        rank=args.rank,
        time_limit=args.time_limit,
        seed=args.seed,
        strategy="correlation",
        relaxation_restarts=args.relaxation_restarts,
        max_iterations=args.max_iterations,
        two_flip_candidates=256,
        return_info=True,
    )
    elapsed = time.perf_counter() - started
    objective = float(x @ (Q @ x))
    record = {
        "instance": str(args.instance),
        "variables": int(Q.shape[0]),
        "nonzeros": int(Q.nnz),
        "objective": objective,
        "solver_seconds": elapsed,
        "seed": args.seed,
        "rank": args.rank,
        "max_iterations": args.max_iterations,
        "relaxation_restarts": args.relaxation_restarts,
        "successive_stages": info["successive_stages"],
        "optimizer_iterations": info["optimizer_iterations"],
        "objective_evaluations": info["objective_evaluations"],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            x=np.asarray(x, dtype=np.int8),
            metadata=json.dumps(record, sort_keys=True),
        )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
