import numpy as np
import os

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

if __name__ == '__main__':
    rng = np.random.default_rng(seed=8)
    os.makedirs('./instances/sparse/', exist_ok=True)
    n = 60
    Q = np.eye(n)
    for indx, xy in enumerate(rng.permutation(n*(n-1)//2)):
        x = 0
        y = 0
        for _i in range(xy):
            y += 1
            if x < y:
                y = 0
                x += 1

        Q[x, y+1] = Q[y+1,x] = 1.
        Jij, i, j = demake(Q)
        np.savez(f'./instances/sparse/sparse{indx}.npz', Jij=Jij, i=i, j=j)
