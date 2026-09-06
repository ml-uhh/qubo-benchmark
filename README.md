# QUBO benchmark

A QUBO describes an optimization problem using binary decisions: each variable
is either 0 or 1, and the objective assigns a cost to individual choices and
pairs of choices. Scheduling, network design, and many other discrete problems
can be written in this form.

This repository provides more than 1,900 benchmark instances and their recorded
results, a self-contained classical solver, and a solver interface for D-Wave
quantum annealing. Together, they support direct experiments on the same QUBO
problems and comparisons of the returned objective values.

Across all 1,856 instances with comparable results, the classical solver found
a lower objective value in 1,280 cases (69.0%) and the same value in 576 cases
(31.0%). D-Wave did not return a result for 56 additional instances.

## Problem definition and file format

For a matrix \(Q\), every instance asks for

$$
    \min_{x \in \{0,1\}^n} x^\mathsf{T}Qx.
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
  \(L=5,\ldots,15\). This family follows the instances used by Muñoz-Bauza and
  Lidar in [*Scaling Advantage in Approximate Optimization with Quantum
  Annealing*](https://doi.org/10.1103/PhysRevLett.134.160601).

For reproducibility, this repository includes the benchmark instances, solver
code, run instructions, and recorded results used in the comparisons. The MIT
license covers the software; third-party datasets retain their original terms.

## Results

The following table summarizes the best recorded objective values. “Better”
means that the solver returned the lower objective value.

| Instance family | Instances | Classical better | Equal | D-Wave better | No D-Wave result |
| --- | ---: | ---: | ---: | ---: | ---: |
| `compsup` | 480 | 80 | 360 | 0 | 40 |
| `random` | 23 | 13 | 7 | 0 | 3 |
| `s28-qac` | 1,386 | 1,177 | 200 | 0 | 9 |
| `QPLIB` | 23 | 10 | 9 | 0 | 4 |
| **All** | **1,912** | **1,280** | **576** | **0** | **56** |

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

The complete S28/QAC comparison is recorded in
`results/s28-qac/classical_reference_results.csv`, with aggregate counts in
`results/s28-qac/classical_reference_summary.csv`. Each published classical
value is the best result from independent seeded runs; every run had a
three-second solver budget. The per-instance file records the number of
completed runs and campaigns. The signed percentage is defined as
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

## Authors

- [Sören Laue](https://www.inf.uni-hamburg.de/en/inst/ab/ml/people/laue.html), Principal Investigator
- [Tomislav Prusina](https://www.inf.uni-hamburg.de/en/inst/ab/ml/people/prusina.html), PhD Student
- [Mark Blacher](https://www.ti2.uni-jena.de/44/mark-blacher), PhD Student

## Acknowledgements

> The authors gratefully acknowledge the Jülich Supercomputing Centre
> ([https://www.fz-juelich.de/jsc](https://www.fz-juelich.de/jsc)) for funding
> this project by providing computing time on the D-Wave Advantage™ System
> JUPSI through the Jülich UNified Infrastructure for Quantum computing
> (JUNIQ).

## License

The software is released under the [MIT License](LICENSE).
