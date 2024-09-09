import numpy as np
import dimod
import pickle
from pathlib import Path


def load_and_rescale_nominal_bqm(dimensions, rescale=1.0):
    """Load the BQM from file."""

    fn = f'./rbm_saved_bqms/bqm_dims_zephyr_{dimensions[0]}_{dimensions[1]}_{dimensions[2]}.pkl'
    dir = Path(__file__).resolve().parent

    with open(dir.joinpath(fn), 'rb') as f:
        bqm = pickle.load(f)['bqm']

    if rescale != 1.0:
        for (u, v), J in bqm.quadratic.items():
            if J != -2:
                bqm.set_quadratic(u, v, J * rescale)

    assert len(bqm.quadratic.values()) == (dimensions[0] - 1) * (dimensions[1] + dimensions[2]) + (
            dimensions[1] * dimensions[2])
    return bqm


def generate_edges(nominal_bqm):
    logical_graph = dimod.to_networkx_graph(nominal_bqm)
    L = list(logical_graph.edges)
    assert L == sorted(L)
    return L


def _bqm(dimensions, periodic, precision, seed):
    rescale = np.round(np.sqrt(4 / dimensions[1]) * 128) / 128
    nominal_bqm = load_and_rescale_nominal_bqm(dimensions)
    edges = list(generate_edges(nominal_bqm))

    # We need to mimic C++ rand() functionality.  Don't use 0 because Matlab can't reproduce it.  Use Mersenne twister
    np.random.seed(seed + 1)
    coin_flips = np.random.randint(256 ** 4, dtype='<u4', size=len(edges))

    if precision == 1:
        coin_flips = coin_flips % 2
        coin_flips = 2 * (coin_flips.astype(int)) - 1
    else:
        # self.precision should be 256 if it isn't 1.
        # Distribute randomly between -128/128 and +128/128, excluding zero.
        myrem = coin_flips % precision
        mysign = 1 - 2 * (myrem >= (precision / 2)).astype(int)
        mymag = (coin_flips % int(precision / 2)) + 1
        coin_flips = mysign * mymag / (precision / 2)

    Js = coin_flips

    # Make the chain couplings -2.  Otherwise rescale.
    Js = Js.astype(float) * rescale
    for i in range(len(Js)):
        if nominal_bqm.quadratic[edges[i]] == -2:
            Js[i] = -2

    bqm = dimod.BQM(vartype='SPIN')
    bqm.add_quadratic_from({(e[0], e[1]): Js[i] for i, e in enumerate(edges)})

    return bqm
