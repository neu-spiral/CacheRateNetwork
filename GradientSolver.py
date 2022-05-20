import cvxpy as cp
from ProbGenerate import Problem, Demand
from helpers import succFun
import logging, argparse
import copy
import time, random


class GradientSolver:

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

    def Dependencies(self):
        '''
        Generate a dictionary self.dependencies: key: (node, item), value: a list of (demand, path)
        '''
        self.dependencies = {}
        for d in range(len(self.demands)):
            item = self.demands[d].item
            paths = self.demands[d].routing_info['paths']
            for p in self.demands[d].routing_info['paths']:
                path = paths[p]
                x = self.demands[d].query_source
                s = succFun(x, path)
                while s is not None:
                    if (x, item) not in self.dependencies:
                        self.dependencies[(x, item)] = [(d,p)]
                    else:
                        self.dependencies[(x, item)].append((d,p))
                    x = s
                    s = succFun(x, path)

    def obj(self, X, R, Dual):
        ecg = 0.0
        sumrate = 0.0
        flow = {}
        for d in R:
            item = self.demands[d].item
            rate = self.demands[d].rate
            sumrate += rate
            paths = self.demands[d].routing_info['paths']

            for path_id in R[d]:
                path = paths[path_id]
                prob = R[d][path_id]
                x = self.demands[d].query_source
                s = succFun(x, path)
                prodsofar = (1 - prob) * (1 - X[x][item])
                while s is not None:
                    ecg += rate * self.weights[(s, x)] * (1 - prodsofar)

                    # calculate flow over each edge
                    if (s, x) in flow:
                        flow[(s, x)] += prodsofar * rate
                    else:
                        flow[(s, x)] = prodsofar * rate

                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X[x][item])

        lagrangian = ecg
        for e in flow:
            lagrangian -= Dual[e] * (flow[e] - self.bandwidths[e])

        # return ecg / sumrate
        return lagrangian, ecg

    def gradient_X(self, X, R, Dual):

        # def delta_grad(X, R, Dual):
        ZX = {}
        for v in self.graph.nodes():
            ZX[v] = {}
            for i in range(self.catalog_size):
                ZX[v][i] = 0

        for (v, i) in self.dependencies:
            X0 = copy.deepcopy(X)
            X0[v][i] = 0

            for (d, p) in self.dependencies[(v, i)]:
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
                    prodsofar *= (1 - X0[x][item])
                    x = s
                    s = succFun(x, path)

                while s is not None:
                    ZX[v][i] += rate * self.weights[(s, x)] * prodsofar

                    ZX[v][i] += Dual[(s, x)] * prodsofar * rate

                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X0[x][item])

        return ZX

    def gradient_R(self, X, R, Dual):

        ZR = {}
        for d in range(len(self.demands)):
            ZR[d] = {}
            for p in self.demands[d].routing_info['paths']:
                ZR[d][p] = 0

                R0 = copy.deepcopy(R)
                R0[d][p] = 0

                item = self.demands[d].item
                rate = self.demands[d].rate
                paths = self.demands[d].routing_info['paths']

                path = paths[p]
                prob = R0[d][p]
                x = self.demands[d].query_source
                s = succFun(x, path)
                prodsofar = (1 - prob)

                while s is not None:
                    prodsofar *= (1 - X[x][item])
                    ZR[d][p] += rate * self.weights[(s, x)] * prodsofar

                    # calculate flow over each edge along path p
                    ZR[d][p] += Dual[(s, x)] * prodsofar * rate

                    x = s
                    s = succFun(x, path)

        return ZR

    def gradient(self, X, R, Dual):
        ZX = {}
        for v in self.graph.nodes():
            ZX[v] = {}
            for i in range(self.catalog_size):
                X1 = copy.deepcopy(X)
                X1[v][i] = 1
                X0 = copy.deepcopy(X)
                X0[v][i] = 0
                ZX[v][i] = self.obj(X1, R, Dual)[0] - self.obj(X0, R, Dual)[0]
        ZR = {}
        for d in range(len(self.demands)):
            ZR[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R1 = copy.deepcopy(R)
                R1[d][p] = 1
                R0 = copy.deepcopy(R)
                R0[d][p] = 0
                ZR[d][p] = self.obj(X, R1, Dual)[0] - self.obj(X, R0, Dual)[0]
        return ZX, ZR

    def adapt(self, X, R, ZX, ZR, step_size):
        for v in self.graph.nodes():
            for i in range(self.catalog_size):
                X[v][i] += step_size * ZX[v][i]

        for d in range(len(self.demands)):
            for p in self.demands[d].routing_info['paths']:
                R[d][p] += step_size * ZR[d][p]

    def adapt_topK(self, X, DX, stepsize):
        for v in DX:
            for i in DX[v]:
                X[v][i] += stepsize * DX[v][i]

    def alg(self, iterations, Dual):
        pass


class FrankWolfe(GradientSolver):
    def find_max(self, ZX, ZR):
        """
         Solve a linear programing D*Z, given gradient Z
        """
        constr = []

        DX = {}
        for v in self.graph.nodes():
            DX[v] = {}
            for i in range(self.catalog_size):
                DX[v][i] = cp.Variable()
                constr.append(DX[v][i] >= 0.)
                constr.append(DX[v][i] <= 1.)

        DR = {}
        for d in range(len(self.demands)):
            DR[d] = {}
            for p in self.demands[d].routing_info['paths']:
                DR[d][p] = cp.Variable()
                constr.append(DR[d][p] >= 0.)
                constr.append(DR[d][p] <= 1.)

        storage = {}
        for v in self.graph.nodes():
            storage[v] = 0
            for i in range(self.catalog_size):
                storage[v] += DX[v][i]
            constr.append(storage[v] <= self.capacities[v])

        out = {}
        for d in range(len(self.demands)):
            out[d] = 0
            for p in self.demands[d].routing_info['paths']:
                out[d] += (1 - DR[d][p])
            constr.append(out[d] >= 1.0)


        obj = 0
        for v in self.graph.nodes():
            for i in range(self.catalog_size):
                obj += DX[v][i] * ZX[v][i]
        for d in range(len(self.demands)):
            for p in self.demands[d].routing_info['paths']:
                obj += DR[d][p] * ZR[d][p]

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        # print("status:", problem.status)

        for v in self.graph.nodes():
            for i in range(self.catalog_size):
                DX[v][i] = DX[v][i].value
        for d in range(len(self.demands)):
            for p in self.demands[d].routing_info['paths']:
                DR[d][p] = DR[d][p].value

        return DX, DR

    def find_topK(self, Z, capacities):
        """Given the gradient as a matrix Z, find the argument of its top K values in each row. K is the capacity of each row."""

        def topK(row, capacity):
            top_capacity = {}
            row.sort(key = lambda x: x[1], reverse = True)
            K = 0
            while capacity:
                key = row[K][0]
                if capacity >= 1.0:
                    top_capacity[key] = 1.0
                    capacity -= 1.0
                    K += 1
                else:
                    top_capacity[key] = capacity
                    capacity = 0
            return top_capacity

        D = {}
        for row in Z:
            items = list(Z[row].items())
            D[row] = topK(items, capacities[row])
        return D

    def alg(self, iterations, Dual):
        X = {}
        for v in self.graph.nodes():
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        R = {}
        for d in range(len(self.demands)):
            R[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R[d][p] = 0

        self.Dependencies()

        gamma = 1. / iterations

        for t in range(iterations):
            # ZX, ZR = self.gradient(X, R, Dual)
            #
            # DX, DR = self.find_max(ZX, ZR)
            # self.adapt(X, R, DX, DR, gamma)

            ZX2 = self.gradient_X(X, R, Dual)
            ZR2 = self.gradient_R(X, R, Dual)

            DX2 = self.find_topK(ZX2, self.capacities)
            DR2 = self.find_topK(ZR2, self.routing_capacities)

            self.adapt_topK(X, DX2, gamma)
            self.adapt_topK(R, DR2, gamma)

            # obj = self.obj(X, R, Dual)

        return X, R


if __name__ == "__main__":
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
    FW = FrankWolfe(P)
    Dual = {}
    random.seed(1995)
    for e in FW.graph.edges():
        Dual[e] = 10*random.random()
    FW.alg(args.iterations, Dual)



