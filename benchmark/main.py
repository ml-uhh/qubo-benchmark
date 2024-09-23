import pickle
import pandas as pd
import numpy as np
import os, yaml
from collections.abc import Iterator
import networkx as nx

def load_problem(fl: str)-> np.ndarray:
    data = np.load(fl)
    i = data['i']
    j = data['j']
    Jij = data['Jij']
    n = max(np.max(i), np.max(j)) + 1
    Q = np.zeros((n,n))
    Q[i,j] = Jij
    if np.any(Q != Q.T):
        Q = Q + Q.T
        
    return Q


if __name__ == '__main__':
    import argparse
    from pprint import pprint
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--update', action='store_true')
    parser.add_argument('--config', type=str, default='./config.yml')
    args = parser.parse_args()

    with open(args.config) as fl:
        configs = yaml.safe_load(fl)

    from solvers import gurobi_minimize, \
        dwave_minimize, geno_minimize

    time_limit = 3600

    dwave_time = dict(time=1.)

    solv_names = [
        ('gurobi_1s', lambda Q, time_limit: gurobi_minimize(Q, time_limit=1)),
        ('gurobi_10s', lambda Q, time_limit: gurobi_minimize(Q, time_limit=10)),
        ('genosolver', geno_minimize),
        ('gurobi', gurobi_minimize),
        ('dwave', dwave_minimize),
        ('gurobi_dwtime', lambda Q, time_limit: gurobi_minimize(Q, time_limit=dwave_time['time']))
    ]

    if 'gurobi_dwtime' in configs['solvers']:
        configs['solvers'].append('dwave')
    
    solv_names = list(filter(lambda x: x[0] in configs['solvers'], solv_names))

    problems = []
    for instance in configs['instances']:
        subfolder = os.path.join('../instances', instance)
        print(subfolder)
        if subfolder[-4:] == '.npz':
            problems.append(subfolder)
        else:
            for dp, _dn, fls in os.walk(subfolder):
                problems.extend((os.path.join(dp, fl) for fl in fls if fl[-4:] == '.npz'))
    
    for prob in problems:
        (res_path, full_name) = os.path.split(prob)
        Qname = full_name[:-4]
        res_path = os.path.join('../results', res_path[len('../instances/'):])

        for (solname, sol) in solv_names:
            res_file = os.path.join(res_path, f'{Qname}_{solname}.pkl')
            if not args.update and os.path.isfile(res_file):
                continue
            
            Q = load_problem(prob)

            try:
                print(f'Starting\t{solname}\t{Qname}')
                start_time = time.time()
                x = sol(Q, time_limit=time_limit)
                end_time = time.time()
            except (KeyboardInterrupt):
                print('Interupted')
                x = -np.ones(Q.shape[0])
                end_time = time.time()
            dwave_time['time'] = end_time - start_time
            G = nx.from_numpy_array(Q)

            os.makedirs(res_path, exist_ok=True)
            with open(res_file, 'wb') as fl:
                pickle.dump({
                    'task': Qname,
                    'solver': solname,
                    'loss': x@Q@x,
                    'time': end_time - start_time,
                    'x': x,
                    'success': np.all(x >= 0),
                    'bipartite': nx.algorithms.bipartite.is_bipartite(G),
                    'planar': nx.check_planarity(G)[0]
                }, fl)
