import numpy as np

def demake(Q):
    n = Q.shape[0]
    Jij = []
    i = []
    j = []
    for _i in range(n):
        for _j in range(n):
            Jij.append(Q[_i,_j])
            i.append(_i)
            j.append(_j)
    return Jij, i, j

def uniform(n, rng=None):
    if rng is None: rng = np.random.default_rng()
    Q = rng.uniform(size=(n,n))
    Q = (Q + Q.T) / 2
    return Q

def normal(n, rng=None):
    if rng is None: rng = np.random.default_rng()
    Q = rng.normal(size=(n,n))
    Q = (Q + Q.T) / 2
    return Q

def integer(n, hi=3, rng=None):
    if rng is None: rng = np.random.default_rng()
    Q = rng.integers(hi, size=(n,n))
    Q = Q + Q.T
    Q[Q==0] = 1
    return Q


def rademacher(n, rng=None):
    if rng is None: rng = np.random.default_rng()
    Q = 2*(rng.uniform(size=(n,n)) < .5) - 1
    indx = np.triu_indices(n)
    Q[indx] = Q.T[indx]
    return Q

if __name__ == '__main__':
    rng = np.random.default_rng(seed=8)
    Q = integer(64, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rint64.npz', Jij=Jij, i=i, j=j)

    Q = integer(65, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rint65.npz', Jij=Jij, i=i, j=j)

    Q = integer(66, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rint66.npz', Jij=Jij, i=i, j=j)

    Q = normal(65, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rnormal65.npz', Jij=Jij, i=i, j=j)

    Q = normal(66, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rnormal66.npz', Jij=Jij, i=i, j=j)

    Q = uniform(65, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/runiform65.npz', Jij=Jij, i=i, j=j)

    Q = uniform(66, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/runiform66.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(65, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher65.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(66, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher66.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(80, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher80.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(100, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher100.npz', Jij=Jij, i=i, j=j)

    Q = normal(80, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rnormal80.npz', Jij=Jij, i=i, j=j)

    Q = uniform(80, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/runiform80.npz', Jij=Jij, i=i, j=j)

    Q = integer(80, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rint80.npz', Jij=Jij, i=i, j=j)

    Q = integer(100, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rint100.npz', Jij=Jij, i=i, j=j)

    Q = integer(200, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rint200.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(200, rng=rng)
    Jij, i, j= demake(Q)
    np.savez('./problems/rrademacher200.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(150, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher150.npz', Jij=Jij, i=i, j=j)

    Q = rademacher(101, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher101.npz', Jij=Jij, i=i, j=j)
    

    Q = rademacher(199, rng=rng)
    Jij, i, j = demake(Q)
    np.savez('./problems/rrademacher199.npz', Jij=Jij, i=i, j=j)
