import cvxpy as cp
from ProbGenerate import Problem
import logging, argparse
import copy


def succFun(node, path):
    """ The successor of a node in the path.
    """
    if node not in path:
        return None
    i = path.index(node)
    if i + 1 == len(path):
        return None
    else:
        return path[i + 1]


def predFun(node, path):
    """The predecessor of a node in the path.
    """
    if node not in path:
        return None
    i = path.index(node)
    if i - 1 < 0:
        return None
    else:
        return path[i - 1]


class GradientSolver:

    def __init__(self, P):
        self.graph = P.graph
        self.capacities = P.capacities
        self.bandwidths = P.bandwidths
        self.weights = P.weights
        self.demands = P.demands

        self.catalog_size = max([d.item for d in self.demands]) + 1

    def obj(self, X, R):
        ecg = 0.0
        sumrate = 0.0
        for d in R:
            item = d.item
            rate = d.rate
            sumrate += rate
            paths = d.routing_info['paths']

            x = d.query_source

            for path_id in R[d]:
                path = paths[path_id]
                prob = R[d][path_id]

                s = succFun(x, path)
                prodsofar = prob * (1 - X[x][item])
                while s is not None:
                    ecg += rate * self.weights[(s, x)] * (1 - prodsofar)

                    x = s
                    s = succFun(x, path)
                    prodsofar *= 1 - X[x][item]

        return ecg / sumrate

    def gradient(self, X, R):
        Z = {}
        for v in self.graph.nodes():
            Z[v] = {}
            for i in range(self.catalog_size):
                X1 = copy.deepcopy(X)
                X1[v][i] = 1
                X0 = copy.deepcopy(X)
                X0[v][i] = 0
                Z[v][i] = self.obj(X1, R) - self.obj(X0, R)

        for d in self.demands:
            Z[d] = {}
            for p in d.routing_info['paths']:
                R1 = copy.deepcopy(R)
                R1[d][p] = 1
                R0 = copy.deepcopy(R)
                R0[d][p] = 0
                Z[d][p] = self.obj(X, R1) - self.obj(X, R0)
        return Z

    def adapt(self, X, R, Z, step_size):
        for v in self.graph.nodes():
            for i in range(self.catalog_size):
                X[v][i] += step_size * Z[v][i]

        for d in self.demands:
            for p in d.routing_info['paths']:
                R[d][p] += step_size * Z[d][p]

    def alg(self, iterations):
        pass


class ProjectAscent(GradientSolver):

    def project(self, X, R):
        """
        Solve a project given X, R: argmin_(Y,S) ((Y,S)-(X,R))^2
        input: X, R (dictionary)
        return: Y, S (dictionary)
        """
        constr = []

        Y = {}
        for v in self.graph.nodes():
            Y[v] = {}
            for i in range(self.catalog_size):
                Y[v][i] = cp.Variable()
                constr.append(Y[v][i] >= 0.)
                constr.append(Y[v][i] <= 1.)

        S = {}
        for d in self.demands:
            S[d] = {}
            for p in d.routing_info['paths']:
                S[d][p] = cp.Variable()
                constr.append(S[d][p] >= 0.)
                constr.append(S[d][p] <= 1.)

        storage = {}
        for v in self.graph.nodes():
            storage[v] = 0
            for i in range(self.catalog_size):
                storage[v] += Y[v][i]
            constr.append(storage[v] <= self.capacities[v])

        out = {}
        for d in self.demands:
            out[d] = 0
            for p in d.routing_info['paths']:
                out[d] += S[d][p]
            constr.append(out[d] <= 1.0)

        flow = {}
        for d in self.demands:
            rate = d.rate
            paths = d.routing_info['paths']

            x = d.query_source

            for p in paths:
                path = paths[p]
                s = succFun(x, path)
                while s is not None:
                    if (s, x) in flow:
                        flow[(s, x)] += S[d][p] * rate
                    else:
                        flow[(s, x)] = S[d][p] * rate
                    x = s
                    s = succFun(x, path)

        for e in flow:
            constr.append(flow[e] <= self.bandwidths[e])


    def alg(self, iterations):
        X = {}
        for v in range(self.node_size):
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        R = {}
        for d in self.demands:
            R[d] = {}
            for p in d.routing_info['paths']:
                R[d][p] = 0

        for t in range(iterations):
            Z = self.gradient(X, R)
            self.adapt(X, R, Z, 1./(t+1))
            X, R = self.project(X, R)

        return X, R

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz','regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom','servicenetwork'])
    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--query_nodes', default=100, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=2, type=int, help='Maximum capacity per cache')
    parser.add_argument('--max_bandwidth', default=10, type=int, help='Maximum bandwidth per edge')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    dir = "INPUT/"
    input = dir + args.outputfile + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%dbandwidth" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
        args.max_bandwidth)
    P = Problem.unpickle_cls(input)
    logging.info('Read data from ' + input)


