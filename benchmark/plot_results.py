import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import regex
import os
import networkx as nx

if __name__ == '__main__':
    rex = regex.compile(r'(.+)_([^_]+)\.pkl')
    ress = []
    for fl in os.listdir('./results'):
        if match := rex.match(fl):
            nem = match[1]
            sol = match[2]
            with open(os.path.join('./results', fl), 'rb') as fld:
                data = pickle.load(fld)
            ress.append(pd.DataFrame.from_dict({ i: [j] for (i,j) in data.items() }))
            

    res = pd.concat(ress, ignore_index=True)
    print(res)
    print(res.groupby('task')[['bipartite', 'planar']].sum())
