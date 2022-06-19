from ProbGenerate import Problem, Demand
from GradientSolver import FrankWolfe
from helpers import succFun, Dependencies
import logging, argparse
import pickle
import os, time
import copy

class PrimalDual:
    """
    Primal Dual Algorithm for Lagrangian L
    """

    def __init__(self, P):
        self.graph = P.graph
        self.demands = P.demands
        self.bandwidths = P.bandwidths
        self.catalog_size = max([d.item for d in self.demands]) + 1

        self.Dual = {}
        self.Dual_old = {}  # dual variable at the last iteration
        for e in self.graph.edges():
            self.Dual[e] = 0
            self.Dual_old[e] = self.Dual[e]

        self.X = {}
        for v in self.graph.nodes():
            self.X[v] = {}
            for i in range(self.catalog_size):
                self.X[v][i] = 0
        self.R = {}
        for d in range(len(self.demands)):
            self.R[d] = {}
            for p in self.demands[d].routing_info['paths']:
                self.R[d][p] = 0

        self.FW = FrankWolfe(P)

    def DualStep(self, X, R, stepsize):
        # calculate flow over each edge
        flow = {}
        overflow = {}
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
            overflow[e] = flow[e] - self.bandwidths[e]
            # if overflow[e] < 0:
            #     overflow[e] = 0
            # self.Dual[e] += stepsize * overflow[e]
            self.Dual[e] += stepsize * overflow[e]
            # print(flow[e], self.bandwidths[e])
            if self.Dual[e] < 0:
                self.Dual[e] = 0
            # overflow[e] /= self.bandwidths[e]
        return overflow, flow

    # def DualStep_momentum(self, X, R, stepsize):
    #     # calculate flow over each edge
    #     momentum = 0.85
    #     flow = {}
    #     overflow = {}
    #     for d in R:
    #         item = self.demands[d].item
    #         rate = self.demands[d].rate
    #         paths = self.demands[d].routing_info['paths']
    #
    #
    #         for path_id in R[d]:
    #             path = paths[path_id]
    #             prob = R[d][path_id]
    #             x = self.demands[d].query_source
    #             s = succFun(x, path)
    #             prodsofar = (1 - prob) * (1 - X[x][item])
    #
    #             while s is not None:
    #                 if (s, x) in flow:
    #                     flow[(s, x)] += prodsofar * rate
    #                 else:
    #                     flow[(s, x)] = prodsofar * rate
    #                 x = s
    #                 s = succFun(x, path)
    #                 prodsofar *= (1 - X[x][item])
    #
    #     for e in flow:
    #         overflow[e] = flow[e] - self.bandwidths[e]
    #         # if overflow[e] < 0:
    #         #     overflow[e] = 0
    #         # self.Dual[e] += stepsize * overflow[e]
    #         temp_Dual = self.Dual[e] + stepsize * overflow[e] + momentum * (self.Dual[e] - self.Dual_old[e])
    #         if temp_Dual < 0:
    #             temp_Dual = 0
    #         self.Dual_old[e], self.Dual[e] = self.Dual[e], temp_Dual
    #         overflow[e] /= self.bandwidths[e]
    #     return overflow, flow

    def adapt(self, X_new, X_old, smooth):
        '''Adapt solution combined with old solution'''
        for v in X_new:
            for i in X_new[v]:
                X_new[v][i] = smooth * X_new[v][i] + (1 - smooth) * X_old[v][i]
                X_old[v][i] = X_new[v][i]


    def alg(self, iterations, stepsize):
        result = []
        dependencies = Dependencies(self.demands)
        for i in range(iterations):
            X, R = self.FW.alg(iterations=100, Dual=self.Dual, dependencies=dependencies)

            # smooth result
            smooth = 2 / (i + 2)
            # smooth = 1
            self.adapt(X, self.X, smooth)
            self.adapt(R, self.R, smooth)

            overflow, flow = self.DualStep(X, R, stepsize / (i+1)**0.5)

            lagrangian, obj = self.FW.obj(X, R, self.Dual)
            print(i, self.Dual, lagrangian)
            result.append((i, X, R, overflow, flow, lagrangian, obj))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PrimalDual',
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
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--iterations', default=1000, type=int, help='Iterations')
    parser.add_argument('--stepsize', default=50, type=int, help='Stepsize')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    dir = "INPUT%d/" % (args.bandwidth_type + 3)
    input = dir + args.inputfile + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size,
        args.max_capacity, args.bandwidth_coefficient)
    P = Problem.unpickle_cls(input)
    logging.info('Read data from ' + input)
    PD = PrimalDual(P)
    result = PD.alg(args.iterations, args.stepsize)
    # if args.bandwidth_type == 1:
    #     dir = "OUTPUT6/"
    # else:
    #     dir = "OUTPUT%d/" % (args.bandwidth_type+6)
    dir = "OUTPUT%d/" % (args.bandwidth_type + 9)

    if not os.path.exists(dir):
        os.mkdir(dir)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth_%dstepsize" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient, args.stepsize)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)


