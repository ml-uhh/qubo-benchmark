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


from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BQM

token = 'DEV-ea6698e8e6c12cc4cd9a2adc4fe946f31fe99d36'
dwave_sampler = DWaveSampler(token=token)

def dwave_minimize(Q, time_limit=None):
    #sampler = cl.get_solver()
    bqm = BQM.from_qubo(Q)
    try:
        sampleset = EmbeddingComposite(dwave_sampler).sample(bqm, return_embedding=True, num_reads=1000, embedding_parameters=dict(timeout=300))#time_limit))
        num_qubits = sum(len(chain) for chain in sampleset.info['embedding_context']['embedding'].values())
        res = sampleset.lowest().first.sample
        res = np.array([ x[1] for x in sorted(res.items()) ])
        assert np.all((res == 0) | (res == 1))
    except ValueError as e:
        print(e)
        res = np.zeros(Q.shape[0])-1
    return num_qubits, res


if __name__ == '__main__':
    import time
    from pprint import pprint
    problems = []
    subfolder = os.path.join('./instances', '.')
    if subfolder[-4:] == '.npz':
        problems.append(subfolder)
    else:
        for dp, _dn, fls in os.walk(subfolder):
            problems.extend((os.path.join(dp, fl) for fl in fls if fl[-4:] == '.npz'))
    
    for prob in problems:
        (res_path, full_name) = os.path.split(prob)
        Qname = full_name[:-4]
        res_path = os.path.join('./results', res_path[len('./instances/'):])

        solname = 'dwave'
        res_file = os.path.join(res_path, f'{Qname}_{solname}.pkl')
        if os.path.isfile(res_file):
            continue

        print(prob)
        Q = load_problem(prob)

        qub, x = dwave_minimize(Q)
        G = nx.from_numpy_array(Q)

        os.makedirs(res_path, exist_ok=True)
        with open(res_file, 'wb') as fl:
            pickle.dump({
                'task': Qname,
                'solver': solname,
                'loss': x@Q@x,
                'x': x,
                'success': np.all(x >= 0),
                'bipartite': nx.algorithms.bipartite.is_bipartite(G),
                'planar': nx.check_planarity(G)[0],
                'qubits': qub
            }, fl)
            
