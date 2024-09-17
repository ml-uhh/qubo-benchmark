import numpy as np
import dimod
from ._3ddimer import generate_edges


def _bqm(dimensions, periodic, precision, seed):
    return make_bicubic_bqm_from_cubic_bqm(
        dimensions,
        make_nominal_cubic_spin_glass_bqm(dimensions, periodic, precision, seed),
        get_nominal_nodimer_couplings(dimensions, periodic, precision, seed)
    )


def get_nominal_nodimer_couplings(dimensions, periodic, precision, seed):
    """Returns a set of nodimer couplings to write over the dimer pairs in the case of a nodimer experiment."""

    # Make enough seeds to cover logical couplings and nodes, then throw away the logical couplings.
    np.random.seed(seed + 1)

    coin_flips = np.random.randint(
        256 ** 4, dtype='<u4',
        size=len(list(generate_edges(dimensions, periodic))) + np.product(dimensions)
    )

    if precision == 1:
        coin_flips = coin_flips % 2
        coin_flips = 2 * (coin_flips.astype(int)) - 1
    else:
        myrem = coin_flips % precision
        mysign = 1 - 2 * (myrem >= (precision / 2)).astype(int)
        mymag = (coin_flips % int(precision / 2)) + 1
        coin_flips = mysign * mymag / (precision / 2)
        pass

    coin_flips = coin_flips[len(list(generate_edges(dimensions, periodic))):]

    return {p: q for p, q in enumerate(coin_flips)}


def make_bicubic_bqm_from_cubic_bqm(dimensions, _cubic_bqm, _pair_couplings):
    # Get the logical version
    _ret = dimod.BinaryQuadraticModel(vartype='SPIN')

    Lx, Ly, Lz = dimensions

    for u in _pair_couplings:
        _ret.add_quadratic(u, u + Lx * Ly * Lz, _pair_couplings[u])

    # And embed it
    for (u, v), val in _cubic_bqm.quadratic.items():
        if (u - v) % Lx:  # x coupling
            _ret.add_quadratic(u, v, val)
        elif (u - v) % (Lx * Ly):  # y coupling
            _ret.add_quadratic(u + Lx * Ly * Lz, v + Lx * Ly * Lz, val)

        else:  # z coupling
            u, v = sorted((u, v))
            if v - u > Lx * Ly:
                _ret.add_quadratic(v, u + Lx * Ly * Lz, val)
            else:
                _ret.add_quadratic(u, v + Lx * Ly * Lz, val)

    return _ret


def make_nominal_cubic_spin_glass_bqm(dimensions, periodic, precision, seed):
    edges = list(generate_edges(dimensions, periodic))

    # We need to mimic C++ rand() functionality.  Don't use 0 because Matlab can't reproduce it.  Use Mersenne twister
    np.random.seed(seed + 1)
    coin_flips = np.random.randint(256 ** 4, dtype='<u4', size=len(edges))

    if precision == 1:
        coin_flips = coin_flips % 2
        coin_flips = 2 * (coin_flips.astype(int)) - 1
    else:
        # precision should be 256 if it isn't 1.
        # Distribute randomly between -128/128 and +128/128, excluding zero.
        myrem = coin_flips % precision
        mysign = 1 - 2 * (myrem >= (precision / 2)).astype(int)
        mymag = (coin_flips % int(precision / 2)) + 1
        coin_flips = mysign * mymag / (precision / 2)

    Js = coin_flips
    bqm = dimod.BQM(vartype='SPIN')
    bqm.add_quadratic_from({(e[0], e[1]): Js[i] for i, e in enumerate(edges)})

    return bqm
