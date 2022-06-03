from ProbGenerate import Problem, Demand
from Route import OptimalRouting
import logging, argparse, pickle, os
from helpers import succFun, Dependencies
import cvxpy as cp


class Greedy(OptimalRouting):

    def GreedyCache(self, R, dependencies):
        X = {}
        for v in self.graph.nodes():
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        free_capacities = self.capacities.copy()
        cardinality = sum(P.capacities.values())

        while cardinality > 0:

            delta_max = 0
            for (v, i) in dependencies:
                if free_capacities[v] > 0:
                    delta_vi = 0
                    temp = X[v][i]
                    X[v][i] = 0

                    for (d, p) in dependencies[(v, i)]:
                        item = self.demands[d].item
                        rate = self.demands[d].rate
                        paths = self.demands[d].routing_info['paths']

                        path = paths[p]
                        prob = R[d][p]
                        x = self.demands[d].query_source
                        s = succFun(x, path)
                        prodsofar = (1 - prob)

                        # calculate cost and flow after node v
                        while x is not v:
                            prodsofar *= (1 - X[x][item])
                            x = s
                            s = succFun(x, path)

                        while s is not None:
                            delta_vi += rate * self.weights[(s, x)] * prodsofar

                            x = s
                            s = succFun(x, path)
                            prodsofar *= (1 - X[x][item])

                    X[v][i] = temp
                    ''' Record the maximum decrease'''
                    if delta_vi > delta_max:
                        new_element = (v, i)
                        delta_max = delta_vi
            if delta_max <= 0.:
                '''there is no cost reduction by placing items'''
                break
            else:
                '''avoid placing the same item to the node twice'''
                del dependencies[new_element]
                v_new, i_new = new_element

                if free_capacities[v_new] > 1:
                    X[v_new][i_new] = 1
                    free_capacities[v_new] -= 1
                    cardinality -= 1
                else:
                    X[v_new][i_new] = free_capacities[v_new]
                    free_capacities[v_new] = 0
                    cardinality -= free_capacities[v_new]

        return X


class CacheRoute(Greedy):

    def alg(self):
        R = {}
        for d in range(len(self.demands)):
            R[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R[d][p] = 0

        dependencies = Dependencies(self.demands)

        X = self.GreedyCache(R, dependencies)
        R = self.OptimalRoute(X)

        obj = self.obj(X, R)
        print('CacheRoute', obj)

        return (X, R, obj)


class RouteCache(Greedy):

    def alg(self):
        X = {}
        for v in self.graph.nodes():
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        dependencies = Dependencies(self.demands)

        R = self.OptimalRoute(X)
        X = self.GreedyCache(R, dependencies)

        obj = self.obj(X, R)
        print('RouteCache', obj)

        return (X, R, obj)


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
    parser.add_argument('--max_capacity', default=5, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    dir = "INPUT/"
    input = dir + args.inputfile + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size,
        args.max_capacity, args.bandwidth_coefficient)
    P = Problem.unpickle_cls(input)
    logging.info('Read data from ' + input)

    CR = CacheRoute(P)
    result = CR.alg()
    dir = "Greedy/CacheRoute/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)

    RC = RouteCache(P)
    result = RC.alg()
    dir = "Greedy/RouteCache/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(result, f)
