from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit.primitives import BackendSampler as Sampler, BackendEstimator as Estimator
#from qiskit_ibm_runtime import Sampler

from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA

import numpy as np

if __name__ == '__main__':
    # Directly construct QP
    n = 5
    Q = np.random.randint(0, 2, size=(n,n))
    Q = np.minimum(Q + Q.T, 1)
    mod = QuadraticProgram("Random integer Q")
    _x = mod.binary_var_list(n)
    mod.minimize(quadratic=Q)
    print(mod.prettyprint())
    
    #assert False

    #seed = 1234
    #algorithm_globals.random_seed = seed

    token = 'f4e4046404e2f70c1a76f9b6b6f5d497ae2540d4d8852b29d9d246e18d74ca63827159f5781c8e7dbbd00f2cdc76e2a408ddf4da7446b42d48ca622923fb3820'
    service = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q/open/main', token=token)
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=5) #
    #backend = service.backend('ibm_brisbane')
    print(backend)

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qubo_op, offset = mod.to_ising()
    #res = pm.run(qubo_op)
    #print(res)

    #assert False
    
    spsa = COBYLA(maxiter=100)
    sampler = Sampler(backend=backend, options=dict(shots=3, optimization_level=3))
    qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=2)
    algorithm = MinimumEigenOptimizer(qaoa)
    print('Started solving')
    print(qaoa.compute_minimum_eigenvalue(qubo_op))
    #result = algorithm.solve(mod)
    print(result.prettyprint())  # prints solution, x=[1, 0, 1, 0], the cost, fval=4
    
