import numpy as np
import dimod


def generate_edges(dimensions, periodic):
    Lx, Ly, Lz = dimensions

    # Add x
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if x < Lx - 1:
                    yield (x + y * Lx + z * Lx * Ly,
                           (x + 1) + y * Lx + z * Lx * Ly)
                elif x == Lx - 1 and periodic[0]:
                    yield (x + y * Lx + z * Lx * Ly,
                           (0) + y * Lx + z * Lx * Ly)
    # Add y
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if y < Ly - 1:
                    yield (x + y * Lx + z * Lx * Ly,
                           x + (y + 1) * Lx + z * Lx * Ly)
                elif y == Ly - 1 and periodic[1]:
                    yield (x + y * Lx + z * Lx * Ly,
                           x + (0) * Lx + z * Lx * Ly)

    # Add z
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if z < Lz - 1:
                    yield (x + y * Lx + z * Lx * Ly,
                           x + y * Lx + (z + 1) * Lx * Ly)
                elif z == Lz - 1 and periodic[2] and Lz > 2:
                    yield (x + y * Lx + z * Lx * Ly, x + y * Lx + (0) * Lx * Ly)


def _bqm(dimensions, periodic, precision, seed):
    return make_bicubic_bqm_from_cubic_bqm(
        dimensions,
        make_nominal_cubic_spin_glass_bqm(dimensions, periodic, precision, seed)
    )


def make_bicubic_bqm_from_cubic_bqm(dimensions, _cubic_bqm):
    _ret = dimod.BinaryQuadraticModel(vartype='SPIN')

    Lx, Ly, Lz = dimensions[0], dimensions[1], dimensions[2]

    for u in _cubic_bqm.variables:
        _ret.add_quadratic(u, u + Lx * Ly * Lz, -2)

    # And embed it
    for (u, v), val in _cubic_bqm.quadratic.items():
        if (u - v) % Lx:  # x coupling
            _ret.add_quadratic(u, v, val)
        elif (u - v) % (Lx * Ly):  # y coupling
            _ret.add_quadratic(u + Lx * Ly * Lz, v + Lx * Ly * Lz, val)

        else:  # z coupling
            _ret.add_quadratic(u, v + Lx * Ly * Lz, val / 2)
            _ret.add_quadratic(u + Lx * Ly * Lz, v, val / 2)
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
