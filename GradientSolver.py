import cvxpy as cp
from ProbGenerate import Problem, Demand
from helpers import succFun
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

    def gradient_X2(self, X, R, Dual, dependencies):
        """
        Calculate gradient of X using X, R, and dual variable
        """

        ZX = {}
        for v in self.graph.nodes():
            ZX[v] = {}
            for i in range(self.catalog_size):
                ZX[v][i] = 0

        for (v, i) in dependencies:

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
                    ZX[v][i] += rate * self.weights[(s, x)] * prodsofar

                    ZX[v][i] += Dual[(s, x)] * prodsofar * rate

                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X[x][item])

        return ZX

    def gradient_X(self, X, R, Dual, dependencies, cost_e):
        """
        Calculate gradient of X using cost_e obtained by gradient_R
        """

        ZX = {}
        for v in self.graph.nodes():
            ZX[v] = {}
            for i in range(self.catalog_size):
                ZX[v][i] = 0

        for (v, i) in dependencies:
            for (d, p) in dependencies[(v, i)]:
                paths = self.demands[d].routing_info['paths']

                path = paths[p]
                prob = R[d][p]
                x = v

                # calculate cost and flow after node v

                s = succFun(v, path)

                while s is not None:
                    # print((d,p), (s,x))
                    ZX[v][i] += cost_e[(d,p)][(s,x)] / (1-X[v][i]) * (1 - prob)

                    x = s
                    s = succFun(x, path)

        return ZX

    def gradient_R(self, X, R, Dual, cost_e):

        ZR = {}
        for d in range(len(self.demands)):
            ZR[d] = {}
            for p in self.demands[d].routing_info['paths']:
                ZR[d][p] = 0

                item = self.demands[d].item
                rate = self.demands[d].rate
                paths = self.demands[d].routing_info['paths']

                path = paths[p]
                x = self.demands[d].query_source
                s = succFun(x, path)
                prodsofar = 1

                while s is not None:
                    prodsofar *= (1 - X[x][item])
                    cost = rate * self.weights[(s, x)] * prodsofar

                    # calculate flow over each edge along path p
                    flow = Dual[(s, x)] * prodsofar * rate

                    ZR[d][p] = ZR[d][p] + cost + flow

                    cost_e[(d, p)][(s, x)] = cost + flow

                    x = s
                    s = succFun(x, path)

        return ZR

    # def gradient(self, X, R, Dual):
    #     ZX = {}
    #     for v in self.graph.nodes():
    #         ZX[v] = {}
    #         for i in range(self.catalog_size):
    #             X1 = copy.deepcopy(X)
    #             X1[v][i] = 1
    #             X0 = copy.deepcopy(X)
    #             X0[v][i] = 0
    #             ZX[v][i] = self.obj(X1, R, Dual)[0] - self.obj(X0, R, Dual)[0]
    #     ZR = {}
    #     for d in range(len(self.demands)):
    #         ZR[d] = {}
    #         for p in self.demands[d].routing_info['paths']:
    #             R1 = copy.deepcopy(R)
    #             R1[d][p] = 1
    #             R0 = copy.deepcopy(R)
    #             R0[d][p] = 0
    #             ZR[d][p] = self.obj(X, R1, Dual)[0] - self.obj(X, R0, Dual)[0]
    #     return ZX, ZR

    # def find_max(self, ZX, ZR):
    #     """
    #      Solve a linear programing D*Z, given gradient Z
    #     """
    #     constr = []
    #
    #     DX = {}
    #     for v in self.graph.nodes():
    #         DX[v] = {}
    #         for i in range(self.catalog_size):
    #             DX[v][i] = cp.Variable()
    #             constr.append(DX[v][i] >= 0.)
    #             constr.append(DX[v][i] <= 1.)
    #
    #     DR = {}
    #     for d in range(len(self.demands)):
    #         DR[d] = {}
    #         for p in self.demands[d].routing_info['paths']:
    #             DR[d][p] = cp.Variable()
    #             constr.append(DR[d][p] >= 0.)
    #             constr.append(DR[d][p] <= 1.)
    #
    #     storage = {}
    #     for v in self.graph.nodes():
    #         storage[v] = 0
    #         for i in range(self.catalog_size):
    #             storage[v] += DX[v][i]
    #         constr.append(storage[v] <= self.capacities[v])
    #
    #     out = {}
    #     for d in range(len(self.demands)):
    #         out[d] = 0
    #         for p in self.demands[d].routing_info['paths']:
    #             out[d] += (1 - DR[d][p])
    #         constr.append(out[d] >= 1.0)
    #
    #
    #     obj = 0
    #     for v in self.graph.nodes():
    #         for i in range(self.catalog_size):
    #             obj += DX[v][i] * ZX[v][i]
    #     for d in range(len(self.demands)):
    #         for p in self.demands[d].routing_info['paths']:
    #             obj += DR[d][p] * ZR[d][p]
    #
    #     problem = cp.Problem(cp.Maximize(obj), constr)
    #     problem.solve()
    #     # print("status:", problem.status)
    #
    #     for v in self.graph.nodes():
    #         for i in range(self.catalog_size):
    #             DX[v][i] = DX[v][i].value
    #     for d in range(len(self.demands)):
    #         for p in self.demands[d].routing_info['paths']:
    #             DR[d][p] = DR[d][p].value
    #
    #     return DX, DR

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


class FrankWolfe(GradientSolver):

    def alg(self, iterations, Dual, dependencies):
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

        cost_e = {}
        for d in range(len(self.demands)):
            paths = self.demands[d].routing_info['paths']
            for p in self.demands[d].routing_info['paths']:
                cost_e[(d, p)] = {}
                path = paths[p]
                x = self.demands[d].query_source
                s = succFun(x, path)
                while s is not None:
                    cost_e[(d, p)][(s, x)] = 0
                    x = s
                    s = succFun(x, path)

        gamma = 1. / iterations

        for t in range(iterations):
            # ZX, ZR = self.gradient(X, R, Dual)
            #
            # DX, DR = self.find_max(ZX, ZR)
            # self.adapt(X, R, DX, DR, gamma)

            ZR2 = self.gradient_R(X, R, Dual, cost_e)
            ZX2 = self.gradient_X(X, R, Dual, dependencies, cost_e)

            DX2 = self.find_topK(ZX2, self.capacities)
            DR2 = self.find_topK(ZR2, self.routing_capacities)

            self.adapt_topK(X, DX2, gamma)
            self.adapt_topK(R, DR2, gamma)

            # lagrangian, obj = self.obj(X, R, Dual)
            # print(t, lagrangian, obj)

        return X, R

class FrankWolfe_cache(GradientSolver):

    def alg(self, iterations, dependencies, R):
        X = {}
        for v in self.graph.nodes():
            X[v] = {}
            for i in range(self.catalog_size):
                X[v][i] = 0

        Dual = {}
        for e in self.graph.edges():
            Dual[e] = 0

        gamma = 1. / iterations

        for t in range(iterations):
            ZX2 = self.gradient_X2(X, R, Dual, dependencies)
            DX2 = self.find_topK(ZX2, self.capacities)
            self.adapt_topK(X, DX2, gamma)

            lagrangian, obj = self.obj(X, R, Dual)
            print(t, obj)

        return X
