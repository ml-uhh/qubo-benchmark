from quenchlib.instances.generate_instances import InstanceGenerator
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, default='2d', choices=['2d', '3ddimer', '3dnodimer', 'diamond', 'biclique'])
    parser.add_argument('--dimensions', type=int, nargs='+', default=(8,8))
    parser.add_argument('--seed', type=int, choices=range(0,20), default=0)
    args = parser.parse_args()
    
    topology = args.topology
    precision = 256
    dimensions = args.dimensions
    seed = args.seed

    ig = InstanceGenerator(topology, precision, dimensions, seed)
    ig.print_couplings()
    
