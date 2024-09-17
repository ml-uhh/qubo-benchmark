import numpy as np
import dimod
import itertools


def coords_3d_to_diamond(coords):
    gray_z_index = [[1, 0], [2, 3]]
    return [
        (
            x,
            y,
            4 * z + gray_z_index[x % 2][y % 2]
        ) for (x, y, z) in coords
    ]


def coords_diamond_to_3d(coords):
    """coords is a list of 3-tuples."""
    # Should we assert x and y are correct?
    for (x, y, z) in coords:
        assert (x % 2) == int((z % 4) / 2)
        assert (y % 2) == 1 - int(((z + 1) % 4) / 2)
    return [
        (
            x,
            y,
            int(z / 4),
        ) for (x, y, z) in coords
    ]


def diamond_to_int_dict(dimensions):
    coords3d = [(x, y, z) for (y, z, x) in itertools.product(
        range(dimensions[1]),
        range(int(dimensions[2] / 4)),
        range(dimensions[0]),
    )]
    coords_diamond_in_order = coords_3d_to_diamond(coords3d)

    assert coords_diamond_to_3d(coords_diamond_in_order) == coords3d

    return {q: i for i, q in enumerate(coords_diamond_in_order)}


def make_diamond_edges_open(dimensions):

    h, w, d = dimensions
    graycode = [(0, 1), (0, 0), (1, 0), (1, 1)]
    levels = [
        [(2 * x + dx, 2 * y + dy, z) for x in range(h) for y in range(w)]
        for z in range(d)
        for (dx, dy) in (graycode[z % 4],)
    ]
    nodes = set().union(*levels)
    return [
        (u, v)
        for u, v in (
            ((x, y, z), (x + dx, y + dy, z + 1))
            for (x, y, z) in nodes
            for d in (z & 1,)
            for dx, dy in ((d, 1 - d), (-d, d - 1))
        )
        if v in nodes
    ]


def make_diamond_edges(dimensions, periodic):
    h, w, d = dimensions
    if periodic[2]:
        assert d % 4 == 0
        edges = [
            ((x0, y0, z0 % d), (x1, y1, z1 % d))
            for (x0, y0, z0), (x1, y1, z1) in make_diamond_edges_open((h, w, d + 1))
        ]
    else:
        edges = make_diamond_edges_open((h, w, d))

    return [(u, v) for u, v in edges if
            u[0] < dimensions[0] and u[1] < dimensions[1] and v[0] < dimensions[0] and v[1] <
            dimensions[1]]


def generate_edges(dimensions, periodic):
    E = make_diamond_edges(dimensions, periodic)
    Einteger = sorted([sorted((diamond_to_int_dict(dimensions)[u], diamond_to_int_dict(dimensions)[v])) for u, v in E])
    return [tuple(e) for e in Einteger]


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
    bqm.add_quadratic_from({(e[0],e[1]):Js[i] for i, e in enumerate(edges)})

    return bqm