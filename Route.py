from ProbGenerate import Problem, Demand
from helpers import succFun
import cvxpy as cp


class OptimalRouting:

    def __init__(self, P):
        self.graph = P.graph
        self.capacities = P.capacities
        self.bandwidths = P.bandwidths
        self.weights = P.weights
        self.demands = P.demands

        self.catalog_size = max([d.item for d in self.demands]) + 1

    def obj(self, X, R):
        ecg = 0.0
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
                    ecg += rate * self.weights[(s, x)] * (1 - prodsofar)

                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X[x][item])

        # return ecg / sumrate
        return ecg


    def OptimalRoute(self, X):
        """
         Solve a linear programing, given cache X
        """
        constr = []

        R = {}
        for d in range(len(self.demands)):
            R[d] = {}
            for p in self.demands[d].routing_info['paths']:
                R[d][p] = cp.Variable()
                constr.append(R[d][p] >= 0.)
                constr.append(R[d][p] <= 1.)

        out = {}
        for d in range(len(self.demands)):
            out[d] = 0
            for p in self.demands[d].routing_info['paths']:
                out[d] += (1 - R[d][p])
            constr.append(out[d] >= 1.0)

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
                    # calculate flow over each edge
                    if (s, x) in flow:
                        flow[(s, x)] += prodsofar * rate
                    else:
                        flow[(s, x)] = prodsofar * rate

                    x = s
                    s = succFun(x, path)
                    prodsofar *= (1 - X[x][item])
        for e in flow:
            constr.append(flow[e] <= self.bandwidths[e])

        obj = self.obj(X, R)

        problem = cp.Problem(cp.Maximize(obj), constr)
        problem.solve()
        # print("status:", problem.status)

        if problem.status == 'optimal':
            for d in range(len(self.demands)):
                for p in self.demands[d].routing_info['paths']:
                    R[d][p] = R[d][p].value
        else:
            R = None
        return R

