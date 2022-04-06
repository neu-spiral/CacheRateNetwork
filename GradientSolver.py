import cvxpy as cp
from ProbGenerate import Problem, Demand
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

    def obj(self, X, R, Dual):
        ecg = 0.0
        sumrate = 0.0
        for d in R:
            item = self.demands[d].item
            rate = self.demands[d].rate
            sumrate += rate
            paths = self.demands[d].routing_info['paths']

            x = self.demands[d].query_source

            for path_id in R[d]:
                path = paths[path_id]
                prob = R[d][path_id]

                s = succFun(x, path)
                prodsofar = (1 - prob) * (1 - X[x][item])
                while s is not None:
                    ecg += rate * self.weights[(s, x)] * (1 - prodsofar)

                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X[x][item])

        flow = {}
        for d in R:
            item = self.demands[d].item
            rate = self.demands[d].rate
            paths = self.demands[d].routing_info['paths']

            x = self.demands[d].query_source

            for path_id in R[d]:
                path = paths[path_id]
                prob = R[d][path_id]

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
            ecg -= Dual[e] * (flow[e] - self.bandwidths[e])


        return ecg / sumrate

    def gradient(self, X, R, Dual):
        ZX = {}
        for v in self.graph.nodes():
            ZX[v] = {}
            for i in range(self.catalog_size):
                X1 = copy.deepcopy(X)
                X1[v][i] = 1
                X0 = copy.deepcopy(X)
                X0[v][i] = 0
                ZX[v][i] = self.obj(X1, R, Dual) - self.obj(X0, R, Dual)
        ZR = {}
        for d in range(len(self.demands)):
            ZR[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R1 = copy.deepcopy(R)
                R1[d][p] = 1
                R0 = copy.deepcopy(R)
                R0[d][p] = 0
                ZR[d][p] = self.obj(X, R1, Dual) - self.obj(X, R0, Dual)
        return ZX, ZR

    def adapt(self, X, R, ZX, ZR, step_size):
        for v in self.graph.nodes():
            for i in range(self.catalog_size):
                X[v][i] += step_size * ZX[v][i]

        for d in range(len(self.demands)):
            for p in self.demands[d].routing_info['paths']:
                R[d][p] += step_size * ZR[d][p]

    def alg(self, iterations):
        pass


# class ProjectAscent(GradientSolver):
#
#     def project(self, X, R):
#         """
#         Solve a project given X, R: argmin_(Y,S) ((Y,S)-(X,R))^2
#         input: X, R (dictionary)
#         return: Y, S (dictionary)
#         """
#         constr = []
#
#         Y = {}
#         for v in self.graph.nodes():
#             Y[v] = {}
#             for i in range(self.catalog_size):
#                 Y[v][i] = cp.Variable()
#                 constr.append(Y[v][i] >= 0.)
#                 constr.append(Y[v][i] <= 1.)
#
#         S = {}
#         for d in self.demands:
#             S[d] = {}
#             for p in d.routing_info['paths']:
#                 S[d][p] = cp.Variable()
#                 constr.append(S[d][p] >= 0.)
#                 constr.append(S[d][p] <= 1.)
#
#         storage = {}
#         for v in self.graph.nodes():
#             storage[v] = 0
#             for i in range(self.catalog_size):
#                 storage[v] += Y[v][i]
#             constr.append(storage[v] <= self.capacities[v])
#
#         out = {}
#         for d in self.demands:
#             out[d] = 0
#             for p in d.routing_info['paths']:
#                 out[d] += S[d][p]
#             constr.append(out[d] <= 1.0)
#
#         flow = {}
#         for d in self.demands:
#             rate = d.rate
#             paths = d.routing_info['paths']
#
#             x = d.query_source
#
#             for p in paths:
#                 path = paths[p]
#                 s = succFun(x, path)
#                 while s is not None:
#                     if (s, x) in flow:
#                         flow[(s, x)] += S[d][p] * rate
#                     else:
#                         flow[(s, x)] = S[d][p] * rate
#                     x = s
#                     s = succFun(x, path)
#
#         for e in flow:
#             constr.append(flow[e] <= self.bandwidths[e])
#
#         obj = 0
#         for v in self.graph.nodes():
#             for i in range(self.catalog_size):
#                 obj += (X[v][i]-Y[v][i])**2
#         for d in self.demands:
#             for p in d.routing_info['paths']:
#                 obj += (R[d][p]-S[d][p])**2
#
#         problem = cp.Problem(cp.Minimize(obj), constr)
#         problem.solve(solver = cp.MOSEK)
#         print("status:", problem.status)
#
#         for v in self.graph.nodes():
#             for i in range(self.catalog_size):
#                 Y[v][i] = Y[v][i].value()
#         for d in self.demands:
#             for p in d.routing_info['paths']:
#                 S[d][p] = S[d][p].value()
#
#         return Y, S
#
#     def alg(self, iterations):
#         X = {}
#         for v in range(self.node_size):
#             X[v] = {}
#             for i in range(self.catalog_size):
#                 X[v][i] = 0
#
#         R = {}
#         for d in self.demands:
#             R[d] = {}
#             for p in d.routing_info['paths']:
#                 R[d][p] = 0
#
#         for t in range(iterations):
#             Z = self.gradient(X, R)
#             self.adapt(X, R, Z, 1./(t+1))
#             X, R = self.project(X, R)
#
#         return X, R


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
        problem.solve(solver=cp.MOSEK)
        # print("status:", problem.status)

        for v in self.graph.nodes():
            for i in range(self.catalog_size):
                DX[v][i] = DX[v][i].value
        for d in range(len(self.demands)):
            for p in self.demands[d].routing_info['paths']:
                DR[d][p] = DR[d][p].value

        return DX, DR

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

        gamma = 1. / iterations
        for t in range(iterations):
            ZX, ZR = self.gradient(X, R, Dual)
            DX, DR = self.find_max(ZX, ZR)
            self.adapt(X, R, DX, DR, gamma)
            # print(t)

        return X, R




