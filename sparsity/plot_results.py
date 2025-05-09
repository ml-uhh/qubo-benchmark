import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx

def get_scores(df, time_limit=3600):
    df = df.copy()
    times = np.minimum(df['time'], time_limit)
    times[df['relative_loss'] > 0] = time_limit
    df['fixed_tm'] = times
    sh = 10.
    group = df.groupby('solver')
    scores = group['fixed_tm'].apply(lambda x: np.exp(np.sum(np.log(np.maximum(1., x+sh)))/len(x))-sh)
    scores = scores / scores.min()
    scores = scores.reset_index(name='scores').set_index('solver')
    time_med = group['time'].apply('median').reset_index(name='median_time').set_index('solver')
    time_fix_med = group['fixed_tm'].apply('median').reset_index(name='median_fixed_time').set_index('solver')
    solved = group['relative_loss'].apply(lambda x: np.sum(x==0.)/len(x)).reset_index(name='%_solved').set_index('solver')
    rel_med = group['relative_loss'].apply(lambda x: np.median(y) if not (y := x[x>0]).empty else 0.).reset_index(name='rel_loss_med_no0').set_index('solver')
    rel_max = group['relative_loss'].apply('max').reset_index(name='rel_loss_max').set_index('solver')
    fin = pd.concat((scores, time_med, time_fix_med, solved, rel_med, rel_max), axis=1)
    return fin

if __name__ == '__main__':
    ress = []
    for dp, _dn, fls in os.walk('./results'):
        for fl in fls:
            if fl[-4:] == '.pkl':
                with open(os.path.join(dp, fl), 'rb') as fld:
                    data = pickle.load(fld)
                data['folder'] = dp
                ress.append(pd.DataFrame.from_dict({ i: [j] for (i,j) in data.items() }))

    res = pd.concat(ress, ignore_index=True)
    res.loc[res['success'] == False, 'loss'] = float('inf')
    opt = res.groupby('task')['loss'].transform('min')
    rel_loss = res.groupby('task')['loss'].transform(lambda x: (x - x.min()) / (abs(x.min()) + 1))
    res['relative_loss'] = rel_loss
    res = res.sort_values(['task', 'solver'])
    print(res)    


    
    for d in [ '30', '40', '50', '60' ]:
        X = res[res['folder'].str.endswith(d)][['task', 'qubits']].to_numpy(dtype=int)
        plt.plot(X[:,0], X[:,1], '.', label=d)

    plt.legend()
    plt.savefig('./taskxqubits.pdf')
    plt.close()

    for d in [ '30', '40', '50', '60' ]:
        X = res[res['folder'].str.endswith(d)]['qubits'].to_numpy(dtype=int)
        Y = res[res['folder'].str.endswith(d)]['x'].apply(sum)
        indx = np.argsort(X)
        plt.plot(np.array(X)[indx], np.array(Y)[indx], '.', label=d)

    plt.legend()
    plt.savefig('./onesxqubits.pdf')
    plt.close()
