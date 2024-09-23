from gurobi_optimods.qubo import solve_qubo
import gurobipy as gp
import genosolver
import numpy as np
#import cupy as cnp

def gurobi_minimize(Q, time_limit=None):
    res = solve_qubo(Q, verbose=0, time_limit=time_limit)
    x = res.solution
    return x

def gurobi_frac_minimize(Q, time_limit=None):
    m = gp.Model()
    x = m.addMVar(Q.shape[0], lb=0., ub=1.)
    m.setObjective(x@Q@x, gp.GRB.MINIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    m.optimize()

    return x.X.round()

def geno_fg(Q):
    def _fg(x):
        Qx = np.dot(Q,x)
        return np.dot(x,Qx), 2*Qx
    return _fg

def geno_minimize(Q, time_limit=None):
    n = len(Q)
    fg = geno_fg(Q)
    best = np.zeros(n)
    for _i in range(10):
        x0 = np.random.rand(n)
        res = genosolver.minimize(fg, x0, lb=np.zeros_like(x0), ub=np.ones_like(x0), np=np, options={'ls':2, 'max_iter': 200})
        x = res.x
        x = np.round(x)
        if np.dot(x, np.dot(Q, x)) < np.dot(best, np.dot(Q, best)):
            best = x
    return best

"""
import azure.quantum
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit.primitives import BackendSampler as Sampler
import qiskit

connection_string = "SubscriptionId=212d1dac-15dc-4f84-851c-6770cf8695f2;ResourceGroupName=AzureQuantum;WorkspaceName=QuboBenchmar;ApiKey=kNn5YGWI2EaSQ4u4JLbcJAidm9rj48WETWwW7WG1pPuH1LnTWcK11BzrtzHY8t8LL1NJ5U9mwyXOAZQUV433EA;QuantumEndpoint=https://germanywestcentral.quantum.azure.com/;"
workspace = Workspace.from_connection_string(connection_string)

provider = AzureQuantumProvider(workspace)


def azure_minimize(Q, time_limit=None):
    '''
    - ionq.simulator
    - ionq.qpu
    - ionq.qpu.aria-1
    - ionq.qpu.aria-2
    - quantinuum.sim.h1-1sc
    - quantinuum.sim.h1-1e
    - quantinuum.qpu.h1-1
    - rigetti.sim.qvm
    - rigetti.qpu.ankaa-2
    - microsoft.estimator
    '''
    backend = provider.get_backend("ionq.qpu.aria-2")
    n = Q.shape[0]
    mod = QuadraticProgram("Qubo")
    _x = mod.binary_var_list(n)
    mod.minimize(quadratic=Q)
    print(np.all(Q == Q.T))
    print(np.linalg.eigvals(Q))
    spsa = COBYLA(maxiter=3)
    sampler = Sampler(backend=backend, options=dict(shots=3))#, optimization_level=3))
    qaoa = QAOA(sampler=sampler, optimizer=spsa, reps=2)
    algorithm = MinimumEigenOptimizer(qaoa)
    try:
        result = algorithm.solve(mod)._x
    except qiskit.transpiler.exceptions.CircuitTooWideForTarget:
        print('Circuit too wide')
        result = np.zeros(n)-1
    #print(result)
    return result
"""

from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BQM

token = 'DEV-a583c7cbea1c24e2b927d552f7c1f6fe3107d77f'
dwave_sampler = DWaveSampler(token=token)

def dwave_minimize(Q, time_limit=None):
    #sampler = cl.get_solver()
    bqm = BQM.from_qubo(Q)
    try:
        sampleset = EmbeddingComposite(dwave_sampler).sample(bqm, num_reads=1000, embedding_parameters=dict(timeout=300))#time_limit))
        res = sampleset.lowest().first.sample
        res = np.array([ x[1] for x in sorted(res.items()) ])
        assert np.all((res == 0) | (res == 1))
    except ValueError as e:
        print(e)
        res = np.zeros(Q.shape[0])-1
    return res

if __name__ == '__main__':
    pass
