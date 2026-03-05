import numpy as np
import os

def demake(Q):
    n = Q.shape[0]
    Jij = []
    i = []
    j = []
    for _i in range(n):
        for _j in range(n):
            if Q[_i,_j]:
                Jij.append(Q[_i,_j])
                i.append(_i)
                j.append(_j)
    return Jij, i, j

if __name__ == '__main__':
    for foldr in os.listdir('.'):
        if os.path.isdir(f'./{foldr}'):
            os.makedirs(f'../../instances/s28-qac/{foldr}', exist_ok=True)
            for task in os.listdir(f'./{foldr}'):
                if task[-len('.txt'):] == '.txt':
                    with open(f'./{foldr}/{task}', 'r') as fl:
                        data = fl.read().split('\n')
                    i = []
                    j = []
                    Jij = []
                    for row in data:
                        if row == "": continue
                        x, y, c = row.split(' ')
                        i.append(int(x))
                        j.append(int(y))
                        Jij.append(float(c))
                    n = max(max(i), max(j)) + 1
                    Q = np.zeros((n,n))
                    Q[i,j] = Jij
                    Q = 4*Q - 2 * np.diag(np.sum(Q + Q.T, axis=1))
                    Jij,i,j = demake(Q)
                    
                    np.savez_compressed(f'../../instances/s28-qac/{foldr}/{task[:-len(".txt")]}.npz', Jij=Jij, i=i, j=j)

    


