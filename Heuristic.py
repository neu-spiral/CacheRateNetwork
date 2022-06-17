from ProbGenerate import Problem, Demand
import logging, argparse, pickle, os, time
from helpers import succFun, Dependencies
from GradientSolver import FrankWolfe_cache
from Route import OptimalRouting


class Heuristic:

    def __init__(self, P):
        self.graph = P.graph
        self.capacities = P.capacities
        self.bandwidths = P.bandwidths
        self.weights = P.weights
        self.demands = P.demands

        self.catalog_size = max([d.item for d in self.demands]) + 1
        self.routing_capacities = {}
        for d in range(len(self.demands)):
            paths = self.demands[d].routing_info['paths']
            self.routing_capacities[d] = len(paths) - 1

        self.FW = FrankWolfe_cache(P)
        self.route = OptimalRouting(P)

    def alg(self, iterations):
        result = []

        Dual = {}
        for e in self.graph.edges():
            Dual[e] = 0
        dependencies = Dependencies(self.demands)

        R = {}
        for d in range(len(self.demands)):
            R[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R[d][p] = 0

        for t in range(iterations):
            X = self.FW.alg(iterations=100, dependencies=dependencies, R=R)
            R = self.route.OptimalRoute(X)
            if R:
                obj = self.route.obj(X, R)

                logging.info((t, obj))
                result.append((X, R, obj))
            else:
                logging.info('infeasible')
                result.append((X, R, 0))
                break
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Heuristic',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', help='Output file')

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle",
                                 "grid_2d", 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert',
                                 'watts_strogatz', 'regular', 'powerlaw_tree', 'small_world', 'geant',
                                 'abilene', 'dtelekom', 'servicenetwork', 'example1', 'example2', 'abilene2'])
    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=5, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between [1, max_paths]')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--iterations', default=100, type=int, help='Iterations')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    dir = "INPUT%d/" % (args.bandwidth_type)
    input = dir + args.inputfile + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size,
        args.max_capacity, args.bandwidth_coefficient)
    P = Problem.unpickle_cls(input)
    logging.info('Read data from ' + input)

    heuristic = Heuristic(P)
    result = heuristic.alg(args.iterations)
    dir = "Heuristic%d/" % (args.bandwidth_type)
    if not os.path.exists(dir):
        os.mkdir(dir)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)

