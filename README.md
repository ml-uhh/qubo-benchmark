# QUBO benchmark

This repository contains QUBO instances, solver wrappers, and recorded results
used to compare classical optimization methods with D-Wave quantum annealing.

For a matrix $Q$, every instance asks for

$$
\min_{x \in \{0,1\}^n} x^\mathsf{T} Q x.
$$

Lower objective values are better. Instance files use NumPy's compressed NPZ
format with coordinate arrays `i`, `j`, and `Jij`. When the stored coordinate
matrix is not symmetric, the benchmark loader adds its transpose. The objective
is always evaluated using the resulting matrix, without an additional offset.

## Data

The instance families are stored below `instances/`:

- `compsup`: graph-structured instances supplied for the JUPSI benchmark.
- `random`: randomly generated QUBOs.
- `Spin-Glass`: spin-glass instances.
- `QPLIB`: selected QPLIB instances converted to the repository's QUBO format.
- `s28-qac`: 1,386 Sidon-28 spin-glass instances, 126 at each size level
  $L=5,\ldots,15$. This family follows the instances used by Muñoz-Bauza and
  Lidar in [*Scaling Advantage in Approximate Optimization with Quantum
  Annealing*](https://doi.org/10.1103/PhysRevLett.134.160601).

The instance data originate from the cited sources. The MIT license in this
repository covers the software; third-party data remain subject to their
original terms.

## Classical reference solver

`benchmark/simple_solver.py` implements several self-contained QUBO heuristics.
The main reference method is successive low-rank Burer-Monteiro relaxation,
random-hyperplane rounding, variable fixing, and greedy bit-flip polishing. It
accepts dense NumPy matrices and sparse SciPy matrices. The sparse path avoids
forming the dense SDP matrix.

Install the two required Python packages:

```bash
python -m pip install -r requirements-classical.txt
```

Run one instance with a three-second budget:

```bash
python benchmark/run_classical_solver.py \
  instances/s28-qac/15/qac_L15_0.npz \
  --time-limit 3 --seed 0 --output solution.npz
```

The command prints a JSON record containing the objective, solver time, and
basic optimization counters. The optional output file stores the binary vector
and the same metadata.

The S28/QAC comparison is recorded in
`results/s28-qac/classical_reference_results.csv`, with aggregate counts in
`results/s28-qac/classical_reference_summary.csv`. Each published classical
value is the best result from independent seeded runs; every run had a
three-second solver budget. The per-instance file records the number of
completed runs and campaigns. Among 1,377 instances with comparable results,
the classical result was lower on 1,177 and equal on 200; D-Wave did not return
a result on nine additional instances. The signed percentage is defined as
`100 * (classical - D-Wave) / abs(D-Wave)`, so a negative value means that the
classical objective was lower.

### Optional compiled sparse kernel

On Linux, the Burer-Monteiro objective and gradient can be evaluated by the
included C++ kernel:

```bash
g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC \
  benchmark/bm_kernel.cpp -o benchmark/libbm_kernel.so
```

The Python solver detects this library automatically. Without it, the same
calculation uses SciPy sparse matrix multiplication. Set
`QUBO_BM_CPP_KERNEL=0` to force the SciPy path.

## Legacy benchmark wrappers

`benchmark/main.py` contains the original batch runner for Gurobi, Geno, and
D-Wave. These integrations require their respective packages, licenses, and
credentials; see `environment.yml`. D-Wave credentials are read from the
standard Ocean configuration or `DWAVE_API_TOKEN` and are never stored in the
repository.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

The software is released under the [MIT License](LICENSE).
