import numpy as np


class InstanceGenerator:

    def __init__(self,
                 topology: str,
                 precision: int,
                 dimensions: tuple,
                 seed: int,
                 ):
        assert topology in ['2d', '3ddimer', '3dnodimer', 'diamond', 'biclique']
        assert precision in [256, 1]
        assert 0 <= seed < 20

        self.topology = topology
        self.precision = precision
        self.dimensions = dimensions
        self.periodic = self.make_periodic()
        self.check_size()
        self.seed = seed
        self.bqm = self.make_bqm()
        pass

    def check_size(self):
        if self.topology == '2d':
            assert len(self.dimensions) == 2
        elif self.topology in ['3ddimer', '3dnodimer', 'diamond', 'biclique']:
            assert len(self.dimensions) == 3
        if self.topology == 'diamond':
            assert (self.dimensions[2] % 4) == 0, "Diamond Lz must be 0 mod 4 to be periodic."
        if self.topology == 'biclique':
            assert 2 <= self.dimensions[0] <= 3
            assert 4 <= self.dimensions[1] <= 36
            assert self.dimensions[0] == 3 or self.dimensions[1] <= 24
            assert self.dimensions[1] == self.dimensions[2]

    def canonical_mps_ordering(self):
        if self.topology == '2d':
            return np.arange(len(self.bqm))

        if self.topology in ['3ddimer', '3dnodimer']:
            V = np.arange(2 * np.product(self.dimensions))
            Vr = np.reshape(V, (self.dimensions[0], self.dimensions[1], self.dimensions[2], 2))
            varorder = np.transpose(Vr, [3, 2, 1, 0]).ravel()
            ret = np.zeros_like(varorder)
            ret[varorder] = sorted(varorder)
            return ret

        if self.topology == 'diamond':
            return np.arange(len(self.bqm))

        if self.topology == 'biclique':
            chainlength, bsize = self.dimensions[:2]
            varorder = np.concatenate((
                np.arange(0, chainlength * (bsize // 2)),
                np.arange(chainlength * bsize, chainlength * 2 * bsize),
                np.arange(chainlength * (bsize // 2), chainlength * bsize)
            ))
            return varorder

    def make_periodic(self):
        if self.topology == '2d':
            return True, False
        if self.topology == '3ddimer':
            return False, False, self.dimensions[2] > 2
        if self.topology == '3dnodimer':
            return False, False, self.dimensions[2] > 2
        if self.topology == 'diamond':
            return False, False, True
        if self.topology == 'biclique':
            return None

    def make_bqm(self):
        if self.topology == '2d':
            from quenchlib.instances._2d import _bqm
        if self.topology == '3ddimer':
            from quenchlib.instances._3ddimer import _bqm
        if self.topology == '3dnodimer':
            from quenchlib.instances._3dnodimer import _bqm
        if self.topology == 'diamond':
            from quenchlib.instances._diamond import _bqm
        if self.topology == 'biclique':
            from quenchlib.instances._biclique import _bqm
        return _bqm(self.dimensions, self.periodic, self.precision, self.seed)

    def print_couplings(self):
        for edge in sorted(list(self.bqm.quadratic.keys())):
            u, v = sorted(list(edge))
            print(f'{u},{v},{self.bqm.quadratic[edge]}')


"""MAIN PARAMETERS"""
topology = '2d'
precision = 256
dimensions = (4, 4)
seed = 0

#ig = InstanceGenerator(topology, precision, dimensions, seed)
#ig.print_couplings()
#print(f'Canonical MPS ordering takes the spins in the following order:\n {ig.canonical_mps_ordering()}')
