import numpy as np
import os
from scipy.sparse import coo_array
import pandas as pd

if __name__ == '__main__':
    L = list(filter(lambda x: 'precision256' in x, os.listdir('./instances/')))
    for l in L:
        print(l)
        for s in range(20):
            data = np.load(f'./instances/{l}/seed{s:02}.npz')
            i = data['i']
            j = data['j']
            Jij = data['Jij']

            n = np.max(i)
            n = max(np.max(j), n)
            n += 1
            Q = coo_array((Jij, (i,j)),shape=(n,n)).toarray()
            Q = 1/4 * (Q + np.diag(Q@np.ones(n) + Q.T@np.ones(n)))
            Q = coo_array(Q)
            i = Q.row
            j = Q.col
            Jij = Q.data
            np.savez_compressed(f'../problems/{l}_seed{s}.npz', i=i, j=j, Jij=Jij)


    L = os.listdir('./data/instances/generated/')
    for l in L:
        print(l)
        for s in range(20):
            data = pd.read_csv(f'./data/instances/generated/{l}/seed{s}.csv', header=None).to_numpy()
            n = int(np.max(data[:,:2])) + 1
            i = data[:,0].astype(int)
            j = data[:,1].astype(int)
            Jij = data[:,2]
            Q = coo_array((Jij, (i,j)),shape=(n,n)).toarray()
            Q = 1/4 * (Q + np.diag(Q@np.ones(n) + Q.T@np.ones(n)))
            Q = coo_array(Q)
            i = Q.row
            j = Q.col
            Jij = Q.data
            
            np.savez_compressed(f'./benchmark/problems/{l}_seed{s}.npz', i=i, j=j, Jij=Jij)
    
