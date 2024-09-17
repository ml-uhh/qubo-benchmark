import numpy as np
import dimod


def generate_edges(dimensions, periodic):
    L1, L2 = dimensions
    for x in range(L2):
        for y in range(L1):
            if x < L2 - 1:
                yield x * L1 + y, ((x + 1) % L2) * L1 + y

    for x in range(L2):
        for y in range(L1):
            if y < L1 - 1:
                yield x * L1 + y, x * L1 + ((y + 1) % L1)

    if periodic[1]:
        x = L2 - 1
        for y in range(L1):
            yield x * L1 + y, ((x + 1) % L2) * L1 + y

    if periodic[0]:
        y = L1 - 1
        for x in range(L2):
            yield x * L1 + y, x * L1 + ((y + 1) % L1)


def _bqm(dimensions, periodic, precision, seed):
    edges = list(generate_edges(dimensions, periodic))

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
    bqm = dimod.BQM(vartype='SPIN')
    bqm.add_quadratic_from({(e[0], e[1]): Js[i] for i, e in enumerate(edges)})

    return bqm
