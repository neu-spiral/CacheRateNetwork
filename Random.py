from ProbGenerate import Problem, Demand
from Route import OptimalRouting
import logging, argparse, pickle, os
from helpers import succFun, Dependencies
import cvxpy as cp


class Random(OptimalRouting):

    def RandomCache(self, R, dependencies):
        X = {}
        for v in self.graph.nodes():
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        for (v, i) in dependencies:
            for (d, p) in dependencies[(v, i)]:
                prob = R[d][p]
                if prob < 1:
                    X[v][i] = 1
                    break

        for v in X:
            item_total = sum(X[v].values())
            if item_total:
                cache_average = self.capacities[v] / item_total
                for i in X[v]:
                    if X[v][i] > 0:
                        X[v][i] = min(cache_average, 1)

        return X


class CacheRoute(Random):

    def alg(self):
        R = {}
        for d in range(len(self.demands)):
            R[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R[d][p] = 0

        dependencies = Dependencies(self.demands)

        X = self.RandomCache(R, dependencies)
        R = self.OptimalRoute(X)

        if R:
            obj = self.obj(X, R)
            print('CacheRoute', obj)
            return (X, R, obj)
        else:
            print('CacheRoute: infeasible')
            return (X, R, 0)


class RouteCache(Random):

    def alg(self):
        X = {}
        for v in self.graph.nodes():
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        dependencies = Dependencies(self.demands)

        R = self.OptimalRoute(X)
        if R:
            X = self.RandomCache(R, dependencies)

            obj = self.obj(X, R)
            print('RouteCache', obj)
            return (X, R, obj)
        else:
            print('RouteCache: infeasible')
            return (X, R, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile', help='Output file')

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle",
                                 "grid_2d", 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert',
                                 'watts_strogatz', 'regular', 'powerlaw_tree', 'small_world', 'geant',
                                 'abilene', 'dtelekom', 'servicenetwork', 'example1'])
    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=5, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    dir = "INPUT3/"
    input = dir + args.inputfile + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size,
        args.max_capacity, args.bandwidth_coefficient)
    P = Problem.unpickle_cls(input)
    logging.info('Read data from ' + input)

    CR = CacheRoute(P)
    result = CR.alg()
    dir = "Random3/CacheRoute/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)

    RC = RouteCache(P)
    result = RC.alg()
    dir = "Random3/RouteCache/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)
