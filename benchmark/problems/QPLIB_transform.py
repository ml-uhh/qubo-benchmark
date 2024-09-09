import numpy as np
import pandas as pd
import os
from scipy.sparse import coo_matrix

def generate()-> tuple[np.ndarray, np.ndarray]:
    L = os.listdir(os.path.join(os.path.dirname(__file__), 'qplib_mittleman/'))

    for l in L:
        with open(os.path.join(os.path.dirname(__file__), f'qplib_mittleman/{l}'), 'r') as fl:
            words = fl.read().split('\n')
        ob, n, m = words[2:5]
        n = int(n.split('#')[0])
        m = int(m.split('#')[0])
        Q = np.zeros((n,n))
        data = pd.DataFrame([ x.split(' ') for x in words[5:5+m] ]).to_numpy(dtype=float)
        Q[data[:,0].astype(int) - 1, data[:,1].astype(int) - 1] = data[:,2]
        Q = Q
        df = float(words[m+5].split('#')[0])
        k = int(words[m+6].split('#')[0])
        b = np.zeros(n) + df

        data = pd.DataFrame([ x.split(' ') for x in words[7+m:7+m+k] ]).to_numpy(dtype=float)
        b[data[:,0].astype(int) - 1] = data[:,1]

        print(words[7+m+k:])
        
        TQ = Q + np.diag(b)
        
        if ob == 'maximize':
            print(ob)
            TQ = -TQ
        
        co = coo_matrix(TQ)
        i = co.row
        j = co.col
        Jij = co.data
        np.savez_compressed(f'./{l[:-6]}.npz', i=i, j=j, Jij=Jij)
    return 

generate()
