from ProbGenerate import Problem, Demand
from GradientSolver import FrankWolfe, succFun
from helpers import succFun
import logging, argparse
import pickle
import os

class PrimalDual:
    """
    Primal Dual Algorithm for Lagrangian L
    """

    def __init__(self, P):
        self.graph = P.graph
        self.demands = P.demands
        self.bandwidths = P.bandwidths

        self.Dual = {}
        for e in self.graph.edges():
            self.Dual[e] = 0

        self.FW = FrankWolfe(P)

    def DualStep(self, X, R, stepsize):
        # calculate flow over each edge
        flow = {}
        for d in R:
            item = self.demands[d].item
            rate = self.demands[d].rate
            paths = self.demands[d].routing_info['paths']


            for path_id in R[d]:
                path = paths[path_id]
                prob = R[d][path_id]
                x = self.demands[d].query_source
                s = succFun(x, path)
                prodsofar = (1 - prob) * (1 - X[x][item])

                while s is not None:
                    if (s, x) in flow:
                        flow[(s, x)] += prodsofar * rate
                    else:
                        flow[(s, x)] = prodsofar * rate
                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X[x][item])

        for e in flow:
            self.Dual[e] += stepsize * (flow[e] - self.bandwidths[e])
            # print(flow[e], self.bandwidths[e])
            if self.Dual[e] < 0:
                self.Dual[e] = 0

    def alg(self, iterations):
        stepsize = 50
        result = []
        for i in range(iterations):
            X, R = self.FW.alg(iterations=100, Dual=self.Dual)
            if iterations % 50 == 49:
                stepsize = stepsize / 2
            self.DualStep(X, R, stepsize)

            obj = self.FW.obj(X, R, self.Dual)
            print(i, self.Dual, obj)
            result.append((i, self.Dual, obj))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', help='Output file')

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle",
                                 "grid_2d", 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert',
                                 'watts_strogatz', 'regular', 'powerlaw_tree', 'small_world', 'geant',
                                 'abilene', 'dtelekom', 'servicenetwork'])
    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=1, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=0.7, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1/max_paths, 1)')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--iterations', default=100, type=int, help='Iterations')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    dir = "INPUT/"
    input = dir + args.inputfile + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size,
        args.max_capacity, args.bandwidth_coefficient)
    P = Problem.unpickle_cls(input)
    logging.info('Read data from ' + input)
    PD = PrimalDual(P)
    result = PD.alg(args.iterations)
    dir = "OUTPUT2/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    fname = dir + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)


