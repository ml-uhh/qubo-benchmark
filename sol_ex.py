from scipy.sparse import coo_array
import pandas as pd
import numpy as np
import scipy

from all_ex import gurobi_minimize, cvxpy_qubo, load_data

if __name__ == '__main__':
    '''
    i = 0
    data = pd.read_csv('./data/instances/generated/biclique_(2, 8, 8)_precision256/seed' + str(i) + '.csv', header=None)
    data = data.to_numpy()
    n = int(np.max(data[:,:2])+1)
    Q = coo_array((data[:,2], data[:,:2].T.astype(int)), shape=(n,n)).toarray()
    Q = Q + Q.T
    '''
    #Q, X = load_data('00')
    
    n = 10
    W = np.random.randint(-10, 11, size=(n,n))
    O = np.zeros((n,n))
    Q = np.block([ [O, W], [W.T, O] ])
    print(Q)
    

    v = gurobi_minimize(Q)

    X, prob = cvxpy_qubo(Q)
    prob.solve()
    print(f'{X.value = }')
    print(f'{v = }')
