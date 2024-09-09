# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:40:30 2024

@author: Sören
"""

import numpy as np
import cvxpy as cp
from scipy.sparse import coo_array

def fun(v, Q):
    return 0.5*np.dot(v, np.dot(Q, v))


def load_data(s):
    data = np.load('instances/2d_(8, 8)_precision256/seed' + s + '.npz')
    i = data['i']
    j = data['j']
    Jij = data['Jij']

    Q = coo_array((Jij, (i, j)), shape=(64, 64)).toarray()
    Q = Q + Q.T

    samples = np.load('samples/qpu/2d_(8, 8)_precision256/20ns/1000_samples_seed' + s + '.npz')
    #samples = np.load('samples/mps/2d_(8, 8)_precision256/20ns/1000_samples_seed00.npz')
    X = samples['states']
    
    return Q, X


def solve_sdp(Q):
    n = 64
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]

    prob = cp.Problem(cp.Minimize(0.5*cp.trace(Q @ X)),
                      constraints)
    prob.solve()
    val, vec = np.linalg.eig(X.value)
    val[val<=0]=0
    V = vec*np.sqrt(val)
    assert(np.linalg.norm(X.value-np.dot(V, V.T)) < 1E-4)
    
    return V


def f_sol(Q, X):
    fs = []
    for v in X:
        f = fun(v, Q)
        fs.append(f)
        
    fs = np.array(fs)
    return fs

for i in range(20):
    s = f"{i:02}"

    Q, X_q = load_data(s)
    f_q = f_sol(Q, X_q)
    
    V = solve_sdp(Q)
    
    U = np.random.randn(64, 1000)
    X_sdp = np.sign(np.dot(V, U)).T
    
    f_sdp = f_sol(Q, X_sdp)
    
    print(s)
    print('best SDP solution', np.min(f_sdp))
    print('best Q   solution', np.min(f_q))
