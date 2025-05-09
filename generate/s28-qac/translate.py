import numpy as np
import os

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
                    np.savez_compressed(f'../../instances/s28-qac/{foldr}/{task[:-len(".txt")]}.npz', Jij=np.array(Jij), i=np.array(i), j=np.array(j))

    


