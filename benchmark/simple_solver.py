import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

try:
    from bm_kernel import CompiledBmKernel
except ImportError:
    CompiledBmKernel = None


def _as_float_qubo(Q):
    if sparse.issparse(Q):
        return Q.astype(float, copy=False).tocsr()
    return np.asarray(Q, dtype=float)


def _quadratic_value(Q, x):
    return float(x @ (Q @ x))


def _matrix_vector_product(matrix, vector):
    return np.asarray(matrix @ vector).ravel()


def _matrix_column(matrix, index):
    if sparse.issparse(matrix):
        return matrix.getcol(index).toarray().ravel()
    return matrix[:, index]


def _make_compiled_bm_kernel(cost, rank):
    if CompiledBmKernel is None or os.environ.get("QUBO_BM_CPP_KERNEL", "1") == "0":
        return None
    if not Path(__file__).with_name("libbm_kernel.so").exists():
        return None
    return CompiledBmKernel(cost, rank)


def _greedy_polish(Q, symmetric_gradient_matrix, diagonal, x, deadline):
    x = np.asarray(x, dtype=np.int8).copy()
    value = _quadratic_value(Q, x)
    gradient = _matrix_vector_product(symmetric_gradient_matrix, x)

    while time.perf_counter() < deadline:
        directions = 1 - 2 * x
        deltas = directions * gradient + diagonal
        index = int(np.argmin(deltas))
        if deltas[index] >= -1e-12:
            break

        direction = int(directions[index])
        value += float(deltas[index])
        x[index] = 1 - x[index]
        gradient += direction * _matrix_column(
            symmetric_gradient_matrix, index
        )

    return x, value


def _two_flip_polish(
    Q,
    symmetric_gradient_matrix,
    diagonal,
    x,
    deadline,
    candidate_count=64,
):
    """Escape one-flip local minima using promising pairs of bit flips."""
    x, value = _greedy_polish(
        Q, symmetric_gradient_matrix, diagonal, x, deadline
    )
    n = x.size
    if n < 2:
        return x, value

    while time.perf_counter() < deadline:
        gradient = _matrix_vector_product(symmetric_gradient_matrix, x)
        directions = 1 - 2 * x
        deltas = directions * gradient + diagonal
        number_of_candidates = min(int(candidate_count), n)
        candidates = np.argpartition(deltas, number_of_candidates - 1)[
            :number_of_candidates
        ]
        candidate_directions = directions[candidates]
        pair_deltas = deltas[candidates, None] + deltas[None, candidates]
        if sparse.issparse(symmetric_gradient_matrix):
            pair_couplings = symmetric_gradient_matrix[candidates][
                :, candidates
            ].toarray()
        else:
            pair_couplings = symmetric_gradient_matrix[
                np.ix_(candidates, candidates)
            ]
        pair_deltas += (
            candidate_directions[:, None]
            * candidate_directions[None, :]
            * pair_couplings
        )
        np.fill_diagonal(pair_deltas, np.inf)
        flat_index = int(np.argmin(pair_deltas))
        pair_delta = float(pair_deltas.flat[flat_index])
        if pair_delta >= -1e-12:
            break

        first_position, second_position = np.unravel_index(
            flat_index, pair_deltas.shape
        )
        first = int(candidates[first_position])
        second = int(candidates[second_position])
        x[first] = 1 - x[first]
        x[second] = 1 - x[second]
        x, value = _greedy_polish(
            Q, symmetric_gradient_matrix, diagonal, x, deadline
        )

    return x, value


def targeted_three_flip_polish(
    Q,
    x,
    time_limit=1.0,
    candidate_count=96,
    max_rounds=20,
):
    """Search promising triples exactly, followed by exact 1/2-flip polish."""
    Q = np.asarray(Q, dtype=float)
    x = np.asarray(x, dtype=np.int8).copy()
    deadline = time.perf_counter() + max(0.0, float(time_limit))
    symmetric_gradient_matrix = Q + Q.T
    diagonal = np.diag(Q)
    x, value = _two_flip_polish(
        Q, symmetric_gradient_matrix, diagonal, x, deadline,
        candidate_count=x.size,
    )
    n = x.size
    if n < 3:
        return x

    for _ in range(int(max_rounds)):
        if time.perf_counter() >= deadline:
            break
        gradient = symmetric_gradient_matrix @ x
        directions = 1 - 2 * x
        deltas = directions * gradient + diagonal
        count = min(max(3, int(candidate_count)), n)
        candidates = np.argpartition(deltas, count - 1)[:count]
        best_delta = 0.0
        best_triple = None
        for first_position in range(count - 2):
            if time.perf_counter() >= deadline:
                break
            first = int(candidates[first_position])
            for second_position in range(first_position + 1, count - 1):
                second = int(candidates[second_position])
                rest = candidates[second_position + 1:]
                triple_deltas = (
                    deltas[first] + deltas[second] + deltas[rest]
                    + directions[first] * directions[second]
                    * symmetric_gradient_matrix[first, second]
                    + directions[first] * directions[rest]
                    * symmetric_gradient_matrix[first, rest]
                    + directions[second] * directions[rest]
                    * symmetric_gradient_matrix[second, rest]
                )
                position = int(np.argmin(triple_deltas))
                candidate_delta = float(triple_deltas[position])
                if candidate_delta < best_delta:
                    best_delta = candidate_delta
                    best_triple = (first, second, int(rest[position]))
        if best_triple is None or best_delta >= -1e-9:
            break
        for index in best_triple:
            x[index] = 1 - x[index]
        x, value = _two_flip_polish(
            Q, symmetric_gradient_matrix, diagonal, x, deadline,
            candidate_count=n,
        )
    return x


def reactive_tabu_minimize(
    Q,
    initial_x=None,
    target_value=None,
    time_limit=10.0,
    seed=0,
    tenure_scale=1.0,
    return_info=False,
):
    """Reactive one-flip tabu search with aspiration and diversification."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int8)
        info = {"iterations": 0, "improvements": 0, "restarts": 0}
        return (empty, info) if return_info else empty

    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + max(0.0, float(time_limit))
    symmetric_gradient_matrix = Q + Q.T
    diagonal = np.diag(Q)
    if initial_x is None:
        initial_x = rng.integers(0, 2, size=n, dtype=np.int8)
    x = np.asarray(initial_x, dtype=np.int8).copy()
    value = float(x @ Q @ x)
    best_x = x.copy()
    best_value = value
    target = -np.inf if target_value is None else float(target_value)
    gradient = symmetric_gradient_matrix @ x
    tabu_until = np.zeros(n, dtype=np.int64)
    base_tenure = max(4, int(float(tenure_scale) * np.sqrt(n)))
    tenure = base_tenure
    maximum_tenure = max(base_tenure + 2, int(0.2 * n))
    last_seen = {}
    iterations = 0
    improvements = 0
    restarts = 0
    last_improvement = 0
    last_reaction = 0

    while time.perf_counter() < deadline and best_value > target + 1e-9:
        directions = 1 - 2 * x
        deltas = directions * gradient + diagonal
        aspirational = value + deltas < best_value - 1e-9
        allowed = (tabu_until <= iterations) | aspirational
        if np.any(allowed):
            masked = np.where(allowed, deltas, np.inf)
            move = int(np.argmin(masked))
        else:
            move = int(np.argmin(tabu_until))
        move_delta = float(deltas[move])
        direction = int(directions[move])
        x[move] = 1 - x[move]
        value += move_delta
        gradient += direction * symmetric_gradient_matrix[:, move]
        jitter = int(rng.integers(-max(1, tenure // 4), max(2, tenure // 4 + 1)))
        tabu_until[move] = iterations + max(2, tenure + jitter)
        iterations += 1

        if value < best_value - 1e-9:
            best_value = value
            best_x = x.copy()
            improvements += 1
            last_improvement = iterations

        if iterations % max(8, n // 4) == 0:
            key = np.packbits(x).tobytes()
            previous = last_seen.get(key)
            if previous is not None and iterations - previous < 4 * n:
                tenure = min(maximum_tenure, max(tenure + 1, int(1.25 * tenure)))
                last_reaction = iterations
            elif iterations - last_reaction > 8 * n and tenure > base_tenure:
                tenure -= 1
                last_reaction = iterations
            last_seen[key] = iterations
            if len(last_seen) > 4096:
                last_seen.clear()

        if iterations - last_improvement > 20 * n:
            x = best_x.copy()
            strength = min(n, max(3, int(np.sqrt(n)) + restarts % 11))
            indices = rng.choice(n, size=strength, replace=False)
            x[indices] = 1 - x[indices]
            value = float(x @ Q @ x)
            gradient = symmetric_gradient_matrix @ x
            tabu_until.fill(0)
            restarts += 1
            last_improvement = iterations

    info = {
        "iterations": iterations,
        "improvements": improvements,
        "restarts": restarts,
        "final_tenure": tenure,
        "final_x": x.copy(),
        "final_value": float(value),
    }
    return (best_x, info) if return_info else best_x


def path_relink_minimize(Q, start_x, guide_x, time_limit=1.0):
    """Greedily trace a path between two elite solutions and polish it."""
    Q = np.asarray(Q, dtype=float)
    start_x = np.asarray(start_x, dtype=np.int8)
    guide_x = np.asarray(guide_x, dtype=np.int8)
    deadline = time.perf_counter() + max(0.0, float(time_limit))
    symmetric_gradient_matrix = Q + Q.T
    diagonal = np.diag(Q)
    best_x = start_x.copy()
    best_value = float(best_x @ Q @ best_x)

    for origin, destination in ((start_x, guide_x), (guide_x, start_x)):
        if time.perf_counter() >= deadline:
            break
        x = origin.copy()
        value = float(x @ Q @ x)
        gradient = symmetric_gradient_matrix @ x
        remaining = set(np.flatnonzero(x != destination).tolist())
        promising = []
        while remaining and time.perf_counter() < deadline:
            indices = np.fromiter(remaining, dtype=int)
            directions = 1 - 2 * x[indices]
            deltas = directions * gradient[indices] + diagonal[indices]
            position = int(np.argmin(deltas))
            move = int(indices[position])
            direction = int(1 - 2 * x[move])
            value += float(deltas[position])
            x[move] = 1 - x[move]
            gradient += direction * symmetric_gradient_matrix[:, move]
            remaining.remove(move)
            if len(promising) < 12 or value < promising[-1][0]:
                promising.append((value, x.copy()))
                promising.sort(key=lambda item: item[0])
                promising = promising[:12]
        for _, candidate in promising:
            if time.perf_counter() >= deadline:
                break
            polished, polished_value = _two_flip_polish(
                Q, symmetric_gradient_matrix, diagonal, candidate, deadline,
                candidate_count=Q.shape[0],
            )
            if polished_value < best_value:
                best_x, best_value = polished, polished_value
    return best_x


def multistart_greedy_minimize(Q, time_limit=1.0, seed=0):
    """Minimize x.T @ Q @ x over binary x using greedy one-bit flips.

    The solver repeatedly starts from a random binary vector and applies the
    improving bit flip with the largest objective decrease. It returns the best
    local optimum found before the time limit.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")

    n = Q.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)

    limit = max(0.0, float(time_limit))
    deadline = time.perf_counter() + limit
    rng = np.random.default_rng(seed)
    symmetric_gradient_matrix = Q + Q.T
    diagonal = np.diag(Q)

    best_x = np.zeros(n, dtype=np.int8)
    best_value = float(best_x @ Q @ best_x)
    restart = 0

    while restart == 0 or time.perf_counter() < deadline:
        if restart == 0:
            x = np.zeros(n, dtype=np.int8)
        else:
            x = rng.integers(0, 2, size=n, dtype=np.int8)

        x, value = _greedy_polish(
            Q, symmetric_gradient_matrix, diagonal, x, deadline
        )

        if value < best_value:
            best_value = value
            best_x = x.copy()
        restart += 1

    return best_x


def iterated_perturbation_minimize(
    Q,
    initial_x=None,
    target_value=None,
    time_limit=10.0,
    seed=0,
    perturbation_sizes=(3, 5, 8, 12, 20, 32),
    two_flip_candidates=128,
    return_info=False,
):
    """Escape a polished local optimum by repeated random kicks and repolishing."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")

    n = Q.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=int)
        info = {"iterations": 0, "improvements": 0}
        return (empty, info) if return_info else empty

    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + max(0.0, float(time_limit))
    symmetric_gradient_matrix = Q + Q.T
    diagonal = np.diag(Q)
    if initial_x is None:
        initial_x = rng.integers(0, 2, size=n, dtype=np.int8)
    best_x, best_value = _two_flip_polish(
        Q,
        symmetric_gradient_matrix,
        diagonal,
        np.asarray(initial_x, dtype=np.int8),
        deadline,
        candidate_count=two_flip_candidates,
    )
    target = -np.inf if target_value is None else float(target_value)
    current_x = best_x.copy()
    current_value = best_value
    iterations = 0
    improvements = 0
    plateau_moves = 0
    sizes = tuple(max(1, min(n, int(size))) for size in perturbation_sizes)

    while time.perf_counter() < deadline and best_value > target + 1e-9:
        gradient = symmetric_gradient_matrix @ current_x
        directions = 1 - 2 * current_x
        deltas = directions * gradient + diagonal
        zero_moves = np.flatnonzero(np.abs(deltas) <= 1e-12)
        candidate = current_x.copy()
        if zero_moves.size and iterations % 3 != 2:
            index = int(rng.choice(zero_moves))
            candidate[index] = 1 - candidate[index]
            plateau_moves += 1
        else:
            strength = sizes[iterations % len(sizes)]
            indices = rng.choice(n, size=strength, replace=False)
            candidate[indices] = 1 - candidate[indices]
        candidate, value = _two_flip_polish(
            Q,
            symmetric_gradient_matrix,
            diagonal,
            candidate,
            deadline,
            candidate_count=two_flip_candidates,
        )
        if value < best_value - 1e-9:
            best_x = candidate
            best_value = value
            improvements += 1
        if value <= current_value + 1e-9:
            current_x = candidate
            current_value = value
        elif iterations % 31 == 30:
            current_x = best_x.copy()
            current_value = best_value
        iterations += 1

    info = {
        "iterations": iterations,
        "improvements": improvements,
        "plateau_moves": plateau_moves,
        "perturbation_sizes": sizes,
        "two_flip_candidates": int(two_flip_candidates),
    }
    return (best_x, info) if return_info else best_x


def exact_block_neighborhood_minimize(
    Q,
    initial_x=None,
    target_value=None,
    time_limit=10.0,
    seed=0,
    block_sizes=(12, 14, 16, 18),
    candidate_pool_factor=4,
    batch_size=8192,
    return_info=False,
):
    """Optimize selected binary blocks exactly and polish accepted moves.

    Blocks mix variables with small one-flip penalties and strongly coupled
    graph neighbors. Enumerating a block exactly permits coordinated moves that
    ordinary low-order polishing cannot reach.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    n = Q.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int8)
        info = {"blocks": 0, "improvements": 0, "largest_block": 0}
        return (empty, info) if return_info else empty

    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + max(0.0, float(time_limit))
    symmetric_Q = 0.5 * (Q + Q.T)
    symmetric_gradient_matrix = 2.0 * symmetric_Q
    diagonal = np.diag(symmetric_Q)
    x = (
        rng.integers(0, 2, size=n, dtype=np.int8)
        if initial_x is None
        else np.asarray(initial_x, dtype=np.int8).copy()
    )
    x, value = _two_flip_polish(
        symmetric_Q, symmetric_gradient_matrix, diagonal, x, deadline,
        candidate_count=min(n, 192),
    )
    best_x, best_value = x.copy(), float(value)
    target = -np.inf if target_value is None else float(target_value)
    sizes = tuple(sorted({max(2, min(n, int(size))) for size in block_sizes}))
    blocks = 0
    improvements = 0

    while time.perf_counter() < deadline and best_value > target + 1e-9:
        k = sizes[blocks % len(sizes)]
        gradient = symmetric_gradient_matrix @ best_x
        directions = 1 - 2 * best_x
        deltas = directions * gradient + diagonal
        pool_size = min(n, max(k, int(candidate_pool_factor) * k))
        pool = np.argpartition(deltas, pool_size - 1)[:pool_size]

        seed_variable = int(rng.choice(pool))
        selected = [seed_variable]
        available = np.ones(n, dtype=bool)
        available[seed_variable] = False
        while len(selected) < k:
            coupling = np.max(np.abs(symmetric_Q[:, selected]), axis=1)
            coupling[~available] = -np.inf
            # Mostly grow a connected, strongly interacting block; retain some
            # low-delta variables to diversify the neighborhoods.
            if rng.random() < 0.75 and np.isfinite(coupling).any():
                candidates = np.argpartition(coupling, -min(8, available.sum()))[
                    -min(8, available.sum()):
                ]
                candidates = candidates[np.isfinite(coupling[candidates])]
                choice = int(rng.choice(candidates))
            else:
                candidates = pool[available[pool]]
                if candidates.size == 0:
                    candidates = np.flatnonzero(available)
                choice = int(rng.choice(candidates))
            selected.append(choice)
            available[choice] = False

        selected = np.asarray(selected, dtype=int)
        outside_mask = np.ones(n, dtype=bool)
        outside_mask[selected] = False
        outside = np.flatnonzero(outside_mask)
        block_Q = symmetric_Q[np.ix_(selected, selected)]
        linear = 2.0 * symmetric_Q[np.ix_(selected, outside)] @ best_x[outside]
        incumbent_block = best_x[selected]
        incumbent_part = float(
            incumbent_block @ block_Q @ incumbent_block
            + linear @ incumbent_block
        )
        best_part = incumbent_part
        best_block = incumbent_block.copy()
        configurations = 1 << k
        shifts = np.arange(k, dtype=np.uint64)
        for start in range(0, configurations, int(batch_size)):
            if time.perf_counter() >= deadline:
                break
            stop = min(configurations, start + int(batch_size))
            identifiers = np.arange(start, stop, dtype=np.uint64)
            bits = ((identifiers[:, None] >> shifts) & 1).astype(float)
            values = np.einsum("bi,ij,bj->b", bits, block_Q, bits, optimize=True)
            values += bits @ linear
            position = int(np.argmin(values))
            if values[position] < best_part - 1e-9:
                best_part = float(values[position])
                best_block = bits[position].astype(np.int8)

        candidate = best_x.copy()
        candidate[selected] = best_block
        candidate, candidate_value = _two_flip_polish(
            symmetric_Q, symmetric_gradient_matrix, diagonal, candidate,
            deadline, candidate_count=min(n, 192),
        )
        if candidate_value < best_value - 1e-9:
            best_x, best_value = candidate, float(candidate_value)
            improvements += 1
        blocks += 1

    info = {
        "blocks": blocks,
        "improvements": improvements,
        "largest_block": max(sizes),
    }
    return (best_x, info) if return_info else best_x


def lbfgsb_round_minimize(Q, time_limit=1.0, seed=0, max_iterations=200):
    """Minimize a box relaxation with L-BFGS-B and round the result.

    Multiple random starts are attempted until the wall-clock time limit is
    reached. The best rounded binary vector is returned.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")

    n = Q.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)

    class TimeLimitReached(Exception):
        pass

    limit = max(0.0, float(time_limit))
    deadline = time.perf_counter() + limit
    rng = np.random.default_rng(seed)
    symmetric_gradient_matrix = Q + Q.T

    def objective(x):
        return float(x @ Q @ x), symmetric_gradient_matrix @ x

    best_x = np.zeros(n, dtype=np.int8)
    best_value = float(best_x @ Q @ best_x)
    restart = 0

    while restart == 0 or time.perf_counter() < deadline:
        latest_x = rng.random(n)

        def stop_at_deadline(x):
            nonlocal latest_x
            latest_x = x.copy()
            if time.perf_counter() >= deadline:
                raise TimeLimitReached

        try:
            result = minimize(
                objective,
                latest_x,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * n,
                callback=stop_at_deadline,
                options={"maxiter": int(max_iterations), "ftol": 1e-12},
            )
            relaxed_x = result.x
        except TimeLimitReached:
            relaxed_x = latest_x

        x = np.rint(relaxed_x).astype(np.int8)
        value = float(x @ Q @ x)
        if value < best_value:
            best_value = value
            best_x = x
        restart += 1

    return best_x


def _lbfgsb_polish_minimize(
    Q,
    time_limit=1.0,
    seed=0,
    max_iterations=200,
    probabilistic_rounding=False,
):
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")

    n = Q.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)

    class RelaxationTimeLimitReached(Exception):
        pass

    start = time.perf_counter()
    limit = max(0.0, float(time_limit))
    deadline = start + limit
    relaxation_deadline = start + 0.5 * limit
    rng = np.random.default_rng(seed)
    symmetric_gradient_matrix = Q + Q.T
    diagonal = np.diag(Q)

    def objective(x):
        return float(x @ Q @ x), symmetric_gradient_matrix @ x

    candidates = [np.zeros(n, dtype=np.int8)]
    restart = 0
    while restart == 0 or time.perf_counter() < relaxation_deadline:
        latest_x = rng.random(n)

        def stop_relaxation(x):
            nonlocal latest_x
            latest_x = x.copy()
            if time.perf_counter() >= relaxation_deadline:
                raise RelaxationTimeLimitReached

        try:
            result = minimize(
                objective,
                latest_x,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * n,
                callback=stop_relaxation,
                options={"maxiter": int(max_iterations), "ftol": 1e-12},
            )
            relaxed_x = result.x
        except RelaxationTimeLimitReached:
            relaxed_x = latest_x

        candidates.append(np.rint(relaxed_x).astype(np.int8))
        if probabilistic_rounding:
            candidates.append((rng.random(n) < relaxed_x).astype(np.int8))
        restart += 1

    unique_candidates = {}
    for candidate in candidates:
        unique_candidates[np.packbits(candidate).tobytes()] = candidate
    candidates = sorted(
        unique_candidates.values(), key=lambda x: float(x @ Q @ x)
    )

    best_x = candidates[0].copy()
    best_value = float(best_x @ Q @ best_x)
    for candidate in candidates:
        if time.perf_counter() >= deadline:
            break
        x, value = _greedy_polish(
            Q, symmetric_gradient_matrix, diagonal, candidate, deadline
        )
        if value < best_value:
            best_value = value
            best_x = x

    while time.perf_counter() < deadline:
        candidate = rng.integers(0, 2, size=n, dtype=np.int8)
        x, value = _greedy_polish(
            Q, symmetric_gradient_matrix, diagonal, candidate, deadline
        )
        if value < best_value:
            best_value = value
            best_x = x

    return best_x


def lbfgsb_round_polish_minimize(Q, time_limit=1.0, seed=0, max_iterations=200):
    """Run L-BFGS-B, round at 0.5, then apply greedy bit-flip polishing."""
    return _lbfgsb_polish_minimize(
        Q,
        time_limit=time_limit,
        seed=seed,
        max_iterations=max_iterations,
        probabilistic_rounding=False,
    )


def lbfgsb_sample_polish_minimize(Q, time_limit=1.0, seed=0, max_iterations=200):
    """Use relaxed values for randomized rounding, then greedy bit flips."""
    return _lbfgsb_polish_minimize(
        Q,
        time_limit=time_limit,
        seed=seed,
        max_iterations=max_iterations,
        probabilistic_rounding=True,
    )


def _qubo_sdp_cost_matrix(Q):
    """Homogenize a QUBO into an Ising-form SDP cost matrix."""
    Q = _as_float_qubo(Q)
    symmetric_Q = 0.5 * (Q + Q.T)
    n = Q.shape[0]
    ones = np.ones(n)
    linear = _matrix_vector_product(symmetric_Q, ones)
    if os.environ.get("QUBO_BM_DENSE") == "1":
        if sparse.issparse(symmetric_Q):
            symmetric_Q = symmetric_Q.toarray()
        cost = np.empty((n + 1, n + 1), dtype=float)
        cost[:n, :n] = 0.25 * symmetric_Q
        cost[:n, n] = 0.25 * linear
        cost[n, :n] = 0.25 * linear
        cost[n, n] = 0.25 * float(ones @ linear)
        return cost
    quadratic = sparse.csr_matrix(0.25 * symmetric_Q)
    column = sparse.csr_matrix((0.25 * linear).reshape(n, 1))
    constant = sparse.csr_matrix([[0.25 * float(ones @ linear)]])
    return sparse.bmat(
        [[quadratic, column], [column.T, constant]], format="csr"
    )


def _single_bm_relaxation(
    Q, rank, rng, deadline, max_iterations, cost=None, compiled_kernel=None,
    lbfgs_memory=10, return_info=False,
):
    class RelaxationTimeLimitReached(Exception):
        pass

    n = Q.shape[0]
    relaxation_started = time.perf_counter()
    preparation_started = relaxation_started
    if cost is None:
        cost = _qubo_sdp_cost_matrix(Q)
    preparation_time = time.perf_counter() - preparation_started

    def normalized_rows(flat_z):
        z = flat_z.reshape(n + 1, rank)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return norms, z / norms

    objective_evaluations = 0
    optimizer_iterations = 0
    matrix_multiply_time = 0.0
    timed_out = False

    def objective(flat_z):
        nonlocal objective_evaluations, matrix_multiply_time
        objective_evaluations += 1
        if compiled_kernel is not None:
            kernel_started = time.perf_counter()
            value, gradient = compiled_kernel.evaluate(flat_z)
            matrix_multiply_time += time.perf_counter() - kernel_started
            return value, gradient
        norms, y = normalized_rows(flat_z)
        multiply_started = time.perf_counter()
        cost_y = cost @ y
        matrix_multiply_time += time.perf_counter() - multiply_started
        value = float(np.sum(y * cost_y))
        gradient_y = 2.0 * cost_y
        radial = np.sum(gradient_y * y, axis=1, keepdims=True)
        gradient_z = (gradient_y - radial * y) / norms
        return value, gradient_z.ravel()

    initial = rng.normal(size=(n + 1, rank))
    initial /= np.linalg.norm(initial, axis=1, keepdims=True)
    latest_z = initial.ravel()

    def stop_at_deadline(flat_z):
        nonlocal latest_z, optimizer_iterations
        optimizer_iterations += 1
        if time.perf_counter() >= deadline:
            latest_z = flat_z.copy()
            raise RelaxationTimeLimitReached

    try:
        result = minimize(
            objective,
            latest_z,
            method="L-BFGS-B",
            jac=True,
            callback=stop_at_deadline,
            options={
                "maxiter": int(max_iterations),
                "maxcor": int(lbfgs_memory),
                "ftol": 1e-12,
            },
        )
        relaxed_z = result.x
    except RelaxationTimeLimitReached:
        timed_out = True
        relaxed_z = latest_z

    _, y = normalized_rows(relaxed_z)
    value = float(np.sum(y * (cost @ y)))
    info = {
        "optimizer_iterations": int(optimizer_iterations),
        "objective_evaluations": int(objective_evaluations),
        "preparation_time": float(preparation_time),
        "matrix_multiply_time": float(matrix_multiply_time),
        "relaxation_time": float(time.perf_counter() - relaxation_started),
        "timed_out": bool(timed_out),
    }
    return (y, value, info) if return_info else (y, value)


def _rank_concentration(y):
    """Return normalized rank concentration and its Euclidean gradient."""
    number_of_rows = y.shape[0]
    gram = y.T @ y
    concentration = float(np.sum(gram * gram) / number_of_rows**2)
    gradient = 4.0 * (y @ gram) / number_of_rows**2
    return concentration, gradient


def rank_collapse_burer_monteiro_minimize(
    Q,
    rank=10,
    time_limit=10.0,
    seed=0,
    penalty_schedule=None,
    initial_penalty=0.001,
    target_effective_rank=1.05,
    max_penalty_stages=12,
    max_iterations=300,
    rounding_trials=32,
    two_flip_candidates=64,
    return_info=False,
):
    """Drive a low-rank SDP factor toward rank one by continuation."""
    Q = _as_float_qubo(Q)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if int(rank) < 1:
        raise ValueError("rank must be positive")

    n = Q.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=int)
        info = {
            "relaxation_value": 0.0,
            "bm_rank": int(rank),
            "rank_penalties": [] if penalty_schedule is None else list(penalty_schedule),
            "effective_ranks": [1.0],
        }
        return (empty, info) if return_info else empty

    class StageTimeLimitReached(Exception):
        pass

    rank = int(rank)
    fixed_penalties = (
        None
        if penalty_schedule is None
        else tuple(float(value) for value in penalty_schedule)
    )
    number_of_stages = (
        int(max_penalty_stages)
        if fixed_penalties is None
        else len(fixed_penalties)
    )
    start = time.perf_counter()
    limit = max(0.0, float(time_limit))
    deadline = start + limit
    optimization_deadline = start + 0.8 * limit
    rng = np.random.default_rng(seed)
    cost = _qubo_sdp_cost_matrix(Q)
    number_of_rows = n + 1
    if sparse.issparse(cost):
        cost_norm = float(np.sqrt(cost.multiply(cost).sum()))
    else:
        cost_norm = float(np.linalg.norm(cost, ord="fro"))
    cost_scale = max(cost_norm * number_of_rows, 1e-12)
    symmetric_gradient_matrix = Q + Q.T
    diagonal = Q.diagonal() if sparse.issparse(Q) else np.diag(Q)

    initial = rng.normal(size=(number_of_rows, rank))
    initial /= np.linalg.norm(initial, axis=1, keepdims=True)
    current_z = initial.ravel()
    relaxation_values = []
    effective_ranks = []
    applied_penalties = []
    optimizer_iterations = []
    objective_evaluations = []
    binary_candidates = [np.zeros(n, dtype=np.int8)]
    adaptive_penalty = 0.0
    previous_effective_rank = None

    def normalized_rows(flat_z):
        z = flat_z.reshape(number_of_rows, rank)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return norms, z / norms

    for stage in range(number_of_stages):
        if stage > 0 and time.perf_counter() >= optimization_deadline:
            break
        penalty = (
            fixed_penalties[stage]
            if fixed_penalties is not None
            else adaptive_penalty
        )
        remaining = max(0.0, optimization_deadline - time.perf_counter())
        if stage == 0:
            stage_budget = 0.4 * max(0.0, optimization_deadline - start)
        else:
            stages_left = max(1, number_of_stages - stage)
            stage_budget = remaining / stages_left
        stage_deadline = min(
            optimization_deadline, time.perf_counter() + stage_budget
        )
        latest_z = current_z.copy()
        stage_iterations = 0
        stage_evaluations = 0

        def objective(flat_z):
            nonlocal stage_evaluations
            stage_evaluations += 1
            norms, y = normalized_rows(flat_z)
            cost_y = cost @ y
            concentration, concentration_gradient = _rank_concentration(y)
            value = float(np.sum(y * cost_y)) / cost_scale
            value += penalty * (1.0 - concentration)
            gradient_y = 2.0 * cost_y / cost_scale
            gradient_y -= penalty * concentration_gradient
            radial = np.sum(gradient_y * y, axis=1, keepdims=True)
            gradient_z = (gradient_y - radial * y) / norms
            return value, gradient_z.ravel()

        def stop_stage(flat_z):
            nonlocal latest_z, stage_iterations
            stage_iterations += 1
            latest_z = flat_z.copy()
            if time.perf_counter() >= stage_deadline:
                raise StageTimeLimitReached

        try:
            result = minimize(
                objective,
                latest_z,
                method="L-BFGS-B",
                jac=True,
                callback=stop_stage,
                options={"maxiter": int(max_iterations), "ftol": 1e-12},
            )
            current_z = result.x
        except StageTimeLimitReached:
            current_z = latest_z

        optimizer_iterations.append(int(stage_iterations))
        objective_evaluations.append(int(stage_evaluations))

        _, y = normalized_rows(current_z)
        relaxation_values.append(float(np.sum(y * (cost @ y))))
        concentration, _ = _rank_concentration(y)
        effective_rank = 1.0 / max(concentration, 1e-12)
        effective_ranks.append(effective_rank)
        applied_penalties.append(penalty)

        anchor_spins = np.where(y[:-1] @ y[-1] >= 0.0, 1, -1)
        binary_candidates.append(((anchor_spins + 1) // 2).astype(np.int8))
        for _ in range(int(rounding_trials)):
            hyperplane = rng.normal(size=rank)
            signs = np.where(y @ hyperplane >= 0.0, 1, -1)
            spins = signs[:-1] * signs[-1]
            binary_candidates.append(((spins + 1) // 2).astype(np.int8))

        if fixed_penalties is None:
            if effective_rank <= float(target_effective_rank):
                break
            if stage == 0:
                adaptive_penalty = float(initial_penalty)
            else:
                relative_drop = max(
                    0.0,
                    (previous_effective_rank - effective_rank)
                    / max(previous_effective_rank, 1e-12),
                )
                if relative_drop < 0.02:
                    adaptive_penalty *= 2.5
                elif relative_drop > 0.30:
                    adaptive_penalty *= 1.15
                else:
                    adaptive_penalty *= 1.5
                adaptive_penalty = min(adaptive_penalty, 100.0)
            previous_effective_rank = effective_rank

    unique_candidates = {}
    for candidate in binary_candidates:
        unique_candidates[np.packbits(candidate).tobytes()] = candidate
    ordered_candidates = sorted(
        unique_candidates.values(), key=lambda x: _quadratic_value(Q, x)
    )

    best_x = ordered_candidates[0].copy()
    best_value = _quadratic_value(Q, best_x)
    for candidate in ordered_candidates:
        if time.perf_counter() >= deadline:
            break
        polished_x, value = _two_flip_polish(
            Q,
            symmetric_gradient_matrix,
            diagonal,
            candidate,
            deadline,
            candidate_count=two_flip_candidates,
        )
        if value < best_value:
            best_value = value
            best_x = polished_x

    info = {
        "relaxation_value": relaxation_values[0],
        "stage_relaxation_values": relaxation_values,
        "bm_rank": rank,
        "rank_penalties": applied_penalties,
        "effective_ranks": effective_ranks,
        "optimizer_iterations": int(sum(optimizer_iterations)),
        "objective_evaluations": int(sum(objective_evaluations)),
        "iterations_per_stage": optimizer_iterations,
        "evaluations_per_stage": objective_evaluations,
        "target_effective_rank": float(target_effective_rank),
        "rounding_trials": int(rounding_trials),
        "two_flip_candidates": int(two_flip_candidates),
    }
    return (best_x, info) if return_info else best_x


def _reduce_qubo(Q, fixed_mask, fixed_values):
    free = np.flatnonzero(~fixed_mask)
    fixed = np.flatnonzero(fixed_mask)
    if sparse.issparse(Q):
        reduced = Q[free][:, free].tocsr()
    else:
        reduced = Q[np.ix_(free, free)].copy()
    constant = 0.0

    if fixed.size:
        values = fixed_values[fixed]
        if sparse.issparse(Q):
            linear = _matrix_vector_product(Q[free][:, fixed], values)
            linear += _matrix_vector_product(Q[fixed][:, free].T, values)
            reduced = reduced + sparse.diags(linear, format="csr")
            constant = _quadratic_value(Q[fixed][:, fixed], values)
        else:
            linear = Q[np.ix_(free, fixed)] @ values
            linear += Q[np.ix_(fixed, free)].T @ values
            reduced[np.diag_indices_from(reduced)] += linear
            constant = _quadratic_value(Q[np.ix_(fixed, fixed)], values)

    return free, reduced, constant


def successive_burer_monteiro_minimize(
    Q,
    rank=5,
    time_limit=1.0,
    seed=0,
    strategy="correlation",
    fix_fraction=0.10,
    confidence_threshold=0.75,
    relaxation_restarts=3,
    max_stages=12,
    consensus_trials=32,
    max_iterations=35,
    lbfgs_memory=5,
    two_flip_candidates=64,
    return_info=False,
    polish=True,
):
    """Repeatedly solve a low-rank relaxation and fix confident variables."""
    Q = _as_float_qubo(Q)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if strategy not in {"correlation", "consensus"}:
        raise ValueError("strategy must be 'correlation' or 'consensus'")
    if not 0.0 < float(fix_fraction) <= 1.0:
        raise ValueError("fix_fraction must lie in (0, 1]")

    n = Q.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=int)
        info = {
            "relaxation_value": 0.0,
            "bm_rank": int(rank),
            "rounding_strategy": strategy,
            "successive_stages": 0,
        }
        return (empty, info) if return_info else empty

    rank = int(rank)
    start = time.perf_counter()
    limit = max(0.0, float(time_limit))
    deadline = start + limit
    relaxation_deadline = start + 0.8 * limit
    rng = np.random.default_rng(seed)
    fixed_mask = np.zeros(n, dtype=bool)
    fixed_values = np.zeros(n, dtype=np.int8)
    suggested_values = np.zeros(n, dtype=np.int8)
    relaxation_values = []
    fixed_counts = []
    restart_counts = []
    optimizer_iterations = []
    objective_evaluations = []
    preparation_times = []
    matrix_multiply_times = []
    relaxation_times = []
    full_binary_candidates = [np.zeros(n, dtype=np.int8)]

    for stage in range(int(max_stages)):
        if not np.any(~fixed_mask) or time.perf_counter() >= relaxation_deadline:
            break

        free, reduced_Q, constant = _reduce_qubo(Q, fixed_mask, fixed_values)
        stage_preparation_started = time.perf_counter()
        stage_cost = _qubo_sdp_cost_matrix(reduced_Q)
        stage_kernel = _make_compiled_bm_kernel(stage_cost, rank)
        preparation_times.append(
            time.perf_counter() - stage_preparation_started
        )
        stages_left = max(1, int(max_stages) - stage)
        remaining_time = max(0.0, relaxation_deadline - time.perf_counter())
        stage_deadline = time.perf_counter() + remaining_time / stages_left
        stage_relaxations = []
        stage_values = []
        for relaxation_restart in range(int(relaxation_restarts)):
            if relaxation_restart > 0 and time.perf_counter() >= stage_deadline:
                break
            restarts_left = max(
                1, int(relaxation_restarts) - relaxation_restart
            )
            restart_time = max(0.0, stage_deadline - time.perf_counter())
            restart_deadline = time.perf_counter() + restart_time / restarts_left
            y, relaxed_value, relaxation_info = _single_bm_relaxation(
                reduced_Q, rank, rng, restart_deadline, max_iterations,
                cost=stage_cost,
                compiled_kernel=stage_kernel,
                lbfgs_memory=lbfgs_memory,
                return_info=True,
            )
            stage_relaxations.append(y)
            stage_values.append(relaxed_value)
            optimizer_iterations.append(
                relaxation_info["optimizer_iterations"]
            )
            objective_evaluations.append(
                relaxation_info["objective_evaluations"]
            )
            matrix_multiply_times.append(
                relaxation_info["matrix_multiply_time"]
            )
            relaxation_times.append(relaxation_info["relaxation_time"])

        relaxation_values.append(constant + min(stage_values))
        restart_counts.append(len(stage_relaxations))

        if strategy == "correlation":
            correlations = np.stack(
                [y[:-1] @ y[-1] for y in stage_relaxations]
            )
            probabilities = np.mean(correlations >= 0.0, axis=0)
            proposal = (probabilities >= 0.5).astype(np.int8)
            confidence = 2.0 * np.abs(probabilities - 0.5)
            confidence += 1e-9 * np.mean(np.abs(correlations), axis=0)
        else:
            votes = np.zeros(free.size, dtype=float)
            completed_trials = 0
            for y in stage_relaxations:
                for _ in range(int(consensus_trials)):
                    if time.perf_counter() >= relaxation_deadline:
                        break
                    hyperplane = rng.normal(size=rank)
                    signs = np.where(y @ hyperplane >= 0.0, 1, -1)
                    votes += (signs[:-1] * signs[-1] + 1) / 2
                    completed_trials += 1
            completed_trials = max(1, completed_trials)
            probabilities = votes / completed_trials
            proposal = (probabilities >= 0.5).astype(np.int8)
            confidence = 2.0 * np.abs(probabilities - 0.5)

        for y in stage_relaxations:
            anchor_spins = np.where(y[:-1] @ y[-1] >= 0.0, 1, -1)
            reduced_candidates = [((anchor_spins + 1) // 2).astype(np.int8)]
            for _ in range(8):
                hyperplane = rng.normal(size=rank)
                signs = np.where(y @ hyperplane >= 0.0, 1, -1)
                spins = signs[:-1] * signs[-1]
                reduced_candidates.append(((spins + 1) // 2).astype(np.int8))
            for reduced_candidate in reduced_candidates:
                full_candidate = suggested_values.copy()
                full_candidate[fixed_mask] = fixed_values[fixed_mask]
                full_candidate[free] = reduced_candidate
                full_binary_candidates.append(full_candidate)

        suggested_values[free] = proposal
        maximum_to_fix = max(1, int(np.ceil(float(fix_fraction) * free.size)))
        eligible = np.flatnonzero(confidence >= float(confidence_threshold))
        if not eligible.size:
            break
        selected = eligible[np.argsort(confidence[eligible])[-maximum_to_fix:]]
        original_indices = free[selected]
        fixed_values[original_indices] = proposal[selected]
        fixed_mask[original_indices] = True
        fixed_counts.append(int(selected.size))

    candidate = suggested_values
    candidate[fixed_mask] = fixed_values[fixed_mask]
    full_binary_candidates.append(candidate.copy())
    symmetric_gradient_matrix = Q + Q.T
    diagonal = Q.diagonal() if sparse.issparse(Q) else np.diag(Q)
    unique_candidates = {}
    for binary_candidate in full_binary_candidates:
        unique_candidates[np.packbits(binary_candidate).tobytes()] = binary_candidate
    ordered_candidates = sorted(
        unique_candidates.values(), key=lambda x: _quadratic_value(Q, x)
    )

    best_x = ordered_candidates[0].copy()
    best_value = _quadratic_value(Q, best_x)
    pre_polish_objective = best_value
    if polish:
        for binary_candidate in ordered_candidates:
            if time.perf_counter() >= deadline:
                break
            polished_x, value = _two_flip_polish(
                Q,
                symmetric_gradient_matrix,
                diagonal,
                binary_candidate,
                deadline,
                candidate_count=two_flip_candidates,
            )
            if value < best_value:
                best_value = value
                best_x = polished_x

    info = {
        "relaxation_value": relaxation_values[0] if relaxation_values else None,
        "stage_relaxation_values": relaxation_values,
        "bm_rank": rank,
        "rounding_strategy": strategy,
        "successive_stages": len(fixed_counts),
        "fixed_counts": fixed_counts,
        "relaxation_restarts": restart_counts,
        "optimizer_iterations": int(sum(optimizer_iterations)),
        "objective_evaluations": int(sum(objective_evaluations)),
        "iterations_per_relaxation": optimizer_iterations,
        "evaluations_per_relaxation": objective_evaluations,
        "preparation_time": float(sum(preparation_times)),
        "matrix_multiply_time": float(sum(matrix_multiply_times)),
        "relaxation_time": float(sum(relaxation_times)),
        "confidence_threshold": float(confidence_threshold),
        "fix_fraction": float(fix_fraction),
        "two_flip_candidates": int(two_flip_candidates),
        "lbfgs_memory": int(lbfgs_memory),
        "pre_polish_objective": float(pre_polish_objective),
    }
    return (best_x, info) if return_info else best_x


def burer_monteiro_minimize(
    Q,
    rank=5,
    time_limit=1.0,
    seed=0,
    max_iterations=200,
    rounding_trials=32,
    return_info=False,
):
    """Solve a low-rank SDP heuristic, then round and greedily polish.

    The SDP variable is represented as X = Y Y.T with unit-norm rows of Y.
    Writing each row as Y_i = Z_i / ||Z_i|| absorbs the sphere constraints
    into the parametrization and permits unconstrained optimization over Z.
    """
    Q = _as_float_qubo(Q)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if int(rank) < 1:
        raise ValueError("rank must be positive")

    n = Q.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=int)
        info = {"relaxation_value": 0.0, "bm_rank": int(rank)}
        return (empty, info) if return_info else empty

    class RelaxationTimeLimitReached(Exception):
        pass

    rank = int(rank)
    start = time.perf_counter()
    limit = max(0.0, float(time_limit))
    deadline = start + limit
    optimization_deadline = start + 0.7 * limit
    rng = np.random.default_rng(seed)
    cost = _qubo_sdp_cost_matrix(Q)
    symmetric_gradient_matrix = Q + Q.T
    diagonal = Q.diagonal() if sparse.issparse(Q) else np.diag(Q)
    relaxed_candidates = []

    def normalized_rows(flat_z):
        z = flat_z.reshape(n + 1, rank)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return z, norms, z / norms

    def objective(flat_z):
        _, norms, y = normalized_rows(flat_z)
        cost_y = cost @ y
        value = float(np.sum(y * cost_y))
        gradient_y = 2.0 * cost_y
        radial = np.sum(gradient_y * y, axis=1, keepdims=True)
        gradient_z = (gradient_y - radial * y) / norms
        return value, gradient_z.ravel()

    restart = 0
    while restart == 0 or time.perf_counter() < optimization_deadline:
        initial = rng.normal(size=(n + 1, rank))
        initial /= np.linalg.norm(initial, axis=1, keepdims=True)
        latest_z = initial.ravel()

        def stop_relaxation(flat_z):
            nonlocal latest_z
            latest_z = flat_z.copy()
            if time.perf_counter() >= optimization_deadline:
                raise RelaxationTimeLimitReached

        try:
            result = minimize(
                objective,
                latest_z,
                method="L-BFGS-B",
                jac=True,
                callback=stop_relaxation,
                options={"maxiter": int(max_iterations), "ftol": 1e-12},
            )
            relaxed_z = result.x
        except RelaxationTimeLimitReached:
            relaxed_z = latest_z

        _, _, y = normalized_rows(relaxed_z)
        relaxed_value = float(np.sum(y * (cost @ y)))
        relaxed_candidates.append((relaxed_value, y.copy()))
        restart += 1

    relaxed_candidates.sort(key=lambda item: item[0])
    best_relaxation_value = relaxed_candidates[0][0]
    binary_candidates = [np.zeros(n, dtype=np.int8)]

    for _, y in relaxed_candidates:
        anchor = y[-1]
        deterministic_spins = np.where(y[:-1] @ anchor >= 0.0, 1, -1)
        binary_candidates.append(((deterministic_spins + 1) // 2).astype(np.int8))

        for _ in range(int(rounding_trials)):
            if time.perf_counter() >= deadline:
                break
            hyperplane = rng.normal(size=rank)
            signs = np.where(y @ hyperplane >= 0.0, 1, -1)
            spins = signs[:-1] * signs[-1]
            binary_candidates.append(((spins + 1) // 2).astype(np.int8))

    unique_candidates = {}
    for candidate in binary_candidates:
        unique_candidates[np.packbits(candidate).tobytes()] = candidate
    binary_candidates = sorted(
        unique_candidates.values(), key=lambda x: _quadratic_value(Q, x)
    )

    best_x = binary_candidates[0].copy()
    best_value = _quadratic_value(Q, best_x)
    for candidate in binary_candidates:
        if time.perf_counter() >= deadline:
            break
        x, value = _greedy_polish(
            Q, symmetric_gradient_matrix, diagonal, candidate, deadline
        )
        if value < best_value:
            best_value = value
            best_x = x

    info = {
        "relaxation_value": best_relaxation_value,
        "bm_rank": rank,
        "rounding_trials": int(rounding_trials),
    }
    return (best_x, info) if return_info else best_x
