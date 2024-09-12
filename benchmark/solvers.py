from autograd import value_and_grad
import autograd.numpy as np
import cvxpy as cp
from gurobi_optimods.qubo import solve_qubo
import gurobipy as gp
import genosolver
#import cupy as cnp

def is_to_bin(Q, b):
    c = 1/2*(Q@np.ones(Q.shape[0]) + b)
    R = (1/4*Q + np.diag(c))
    return R

def gurobi_minimize(Q, time_limit=None):
    res = solve_qubo(Q, verbose=0, time_limit=time_limit)
    x = res.solution
    return x

def random_cut(V):
    u = np.random.randn(V.shape[1])
    z = np.sign(V @ u)
    z[z==0] = 1
    return z

import scipy

def cvxpy_qubo(Q):
    X = cp.Variable(Q.shape, symmetric=True)
    constraints = [X >>0 , cp.diag(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints)
    return X, prob

def cvxpy_minimize(Q, time_limit=None):
    X, prob = cvxpy_qubo(Q)
    if time_limit is not None:
        prob.solve(cplex_params={"timelimit": time_limit})
    else:
        prob.solve()
    U, s, _U = np.linalg.svd(X.value, hermitian=True)
    V = U*np.sqrt(s)[...,None,:]
    best = np.ones(Q.shape[0])
    for _i in range(1000):
        curr = random_cut(V)
        if best @ Q @ best > curr @ Q @ curr:
            best = curr
    return best

def g_qubo(Q):
    def _g(x):
        V = x.reshape(Q.shape[0],-1)
        V1 = np.sum(V**2, axis=1)
        Vb = np.outer(V1,V1)
        return 2*((Q*Vb**-0.5) - np.diag((V@V.T*Q*Vb**-1.5)@V1))@V
    return _g

def fg_qubo(Q, lam=0.):
    def _fg(x):
        V = x.reshape(Q.shape[0], -1)
        V1 = np.sum(V*V, axis=1)
        U = V@V.T
        Vb = np.outer(V1,V1)
        X = U*(Vb**-0.5)
        XQ = X*Q

        f = np.sum(XQ)
        g = 2*((Q*Vb**-0.5) - np.diag((XQ*(Vb**-1))@V1))@V
        return f, g.reshape(x.shape)
    return _fg

def geno_minimize(Q, time_limit=None):
    n = len(Q)
    x0 = np.random.randn(n,n).reshape(-1)
    for lam in [ 0.]:# 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.05, 0.1, 1., 4., 16., 32. ]:
        fg = fg_qubo(Q, lam=lam)
        res = genosolver.minimize(fg, x0, np=np)
        x0 = res.x
    x = x0.reshape(n,n)
    V = x/np.sqrt(np.sum(x*x, axis=1))[:,None]
    X = V.dot(V.T)
    best = np.ones(Q.shape[0])
    for _i in range(1000):
        curr = random_cut(V)
        if best @ Q @ best > curr @ Q @ curr:
            best = curr
    z = best
    #print(f'{np.sum(X * Q) = }')
    #z = np.sign(V[:,0])
    #z[z==0] = 1.
    return z

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


from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BQM

token = 'DEV-42a3b7ec68e6c9a8978cc4a1ff7c3051b87cd1e5'    
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
