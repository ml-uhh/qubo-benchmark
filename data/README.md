# Computational supremacy in quantum simulation: Initial data repository

This data repository is intended to support the findings of the research article 'Computational supremacy in quantum simulation'.

## Contents

### Instances

- Instance generators are provided in `src`.  Some pathing and import adjustments may be necessary.
- A selection of small instances is also provided as `.npz` files, with the self-explanatory fields `'i'`, `'j'`, and `'Jij'`.
- **Caution**: The numbering of sites gives a good MPS ordering for 2D instances, but not in general.  Appropriate MPS orderings can be found in the instance generation class in `src/quenchlib/instances/generate_instances.py`.


### Correlations

- Spin-spin correlations are given for both MPS (at a range of bond dimensions chi) and QPU.
- Each `.npz` file contains a matrix with N-choose-2 two-body correlations for each of 20 seeds (0 to 19).  Correlations are given in row-wise upper-triangular order: (0,1), (0,2), ... (0,N-1), (1,2), (1,3), ..., (1,N-1), ..., (N-2,N-1).
- ADV1 correlations are only given for square inputs as in Fig. 2D and Fig. 2E.
- Most correlations are given for t_a=7ns and 20ns.  8x8 correlations are also given for 1ns, 2ns, and 4ns.

### Samples

- 1000 samples are provided for the instances and quench times shown in Fig. 5A and Fig. 5B (inset).  They can be used to estimate <q^2>, U, and other observables.

### Quench annealing schedule

- The model annealing schedule (based on the ADV1 QPU) is given in `src/quenchlib/schedules/ADV1.csv`. 
- This file erroneously contained columns s, 2Γ, 2J in the initial version, and now contains columns s, Γ, J.
- The schedule uses the reduced Planck constant ℏ, so it may be necessary to multiply Γ and J by 2π to reproduce expected results.


## Contact

Andrew King (aking@dwavesys.com)

Mohammad Amin (mhsamin@dwavesys.com)