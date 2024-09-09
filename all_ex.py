from autograd import value_and_grad
import autograd.numpy as np
import cvxpy as cp
from gurobi_optimods.qubo import solve_qubo
import gurobipy as gp

#import sys
#sys.path.insert(0, './genosolver')
import genosolver

def brute_minimize(Q):
    n = Q.shape[0]
    best = np.ones(n)
    for i in range(2**n):
        b = np.binary_repr(i, width=n)
        x = np.array([ 2*int(bi) - 1. for bi in b ])
        if best @ Q @ best > x @ Q @ x:
            best = x
    return best

def gurobi_minimize(Q):
    R = 4 * (Q - np.diag(Q @ np.ones(Q.shape[0])))
    res = solve_qubo(R, verbose=0)
    x = res.solution
    x = 2 * np.array(x) - 1
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

def cvxpy_minimize(Q):
    X, prob = cvxpy_qubo(Q)
    prob.solve()
    
    U, s, _U = np.linalg.svd(X.value, hermitian=True)
    V = U*s[...,None,:]
    best = np.ones(Q.shape[0])
    for _i in range(1000):
        curr = random_cut(V)
        if best @ Q @ best > curr @ Q @ curr:
            best = curr
    return best

def fg_qubo(Q, lam=0.):
    nu = 0. #if lam==0. else 1e-6
    def _f(x):
        V = x.reshape(Q.shape[0], -1)
        v = np.sqrt(np.sum(V*V, axis=1))
        V2 = V / v[:,None]
        X = np.dot(V2, V2.T)
        return np.sum(X * Q) + lam * np.sum(V2[:,1:]**2) + nu * np.sum(V[:,0]**2)
    return value_and_grad(_f)

def geno_minimize(Q):
    n = len(Q)
    x0 = np.random.randn(n,n).reshape(-1)
    for lam in [ 0.]:# 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.05, 0.1, 1., 4., 16., 32. ]:
        fg = fg_qubo(Q, lam=lam)
        res = genosolver.minimize(fg, x0, np=np)
        x0 = res.x
        print(res)
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

from scipy.sparse import coo_array
import pandas as pd

def load_data(s):
    data = np.load('./data/instances/biclique_(2, 5, 5)_precision256/seed' + s + '.npz')
    i = data['i']
    j = data['j']
    Jij = data['Jij']

    Q = coo_array((Jij, (i, j)), shape=(64, 64)).toarray()
    Q = Q + Q.T

    samples = np.load('./data/samples/qpu/2d_(8, 8)_precision256/20ns/1000_samples_seed' + s + '.npz')
    #samples = np.load('samples/mps/2d_(8, 8)_precision256/20ns/1000_samples_seed00.npz')
    X = samples['states']
    
    '''
    data = pd.read_csv('./data/instances/generated/biclique_(2, 16, 16)_precision256/seed' + str(int(s)) + '.csv', header=None)
    data = data.to_numpy()
    n = int(np.max(data[:,:2])+1)
    Q = coo_array((data[:,2], data[:,:2].T.astype(int)), shape=(n,n)).toarray()
    Q = Q + Q.T
    X = (np.random.randint(2, size=n) + 1)/2
    '''
    return Q, X


if __name__ == '__main__':
    import time
    for i in range(20):
        s = f'{i:02}'
        Q, X_q = load_data(s)
        
        #print(f'{Q = }')
        #print(f'{len(Q) = }')
        
        start_time = time.time()
        x_geno = geno_minimize(Q)
        #print(f'{time.time() - start_time = }')
        print(f'{x_geno @ Q @ x_geno = }')
        #print(f'{x_geno = }')
        #print()
        
        start_time = time.time()
        x_gurobi = gurobi_minimize(Q)
        #print(f'{time.time() - start_time = }')
        print(f'{x_gurobi @ Q @ x_gurobi = }')
        #print(f'{x_gurobi = }')
        #print()

        print(f'{np.min(X_q @ Q @ X_q.T) = }')
        #print(f'{np.max(X_q @ Q @ X_q.T) = }')
        #print()

    
        #start_time = time.time()
        x_cvxpy = cvxpy_minimize(Q)
        #print(f'{time.time() - start_time = }')
        print(f'{x_cvxpy @ Q @ x_cvxpy = }')
        #print(f'{x_cvxpy = }')
        print()

        '''
        start_time = time.time()
        x_brute = brute_minimize(Q)
        print(f'{time.time() - start_time = }')
        print(f'{x_brute @ Q @ x_brute = }')
        print(f'{x_brute = }')
        print()
        '''
