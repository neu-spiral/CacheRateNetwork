from cvxopt import solvers
from cvxopt import matrix
import numpy as np
from scipy.stats import rv_discrete
from networkx import single_source_dijkstra
from networkx import grid_2d_graph
from random import random
import logging


def pp(l):
    return ' '.join(map(str, l))


def scaleDictionary(d, s):
    return dict([(key, s * d[key]) for key in d])


def projectToSimplex(d, cap):
    if len(d) == 0:
        return d, None
    if len(d) == 1:
        for x in d:
            d[x] = cap
        return d, None

    keys, vals = zip(*[(key, d[key]) for key in d])

    n = len(vals)
    q = -matrix(vals)
    P = matrix(np.eye(n))

    G = matrix(np.concatenate((np.eye(n), -np.eye(n), np.ones((1, n)))))

    h = matrix(n * [1.0] + n * [0.0] + [cap])

    solvers.options['show_progress'] = False
    res = solvers.qp(P, q, G, h)

    sol = res['x']
    return dict(zip(keys, sol)), res


def constructDistribution(d, cap):
    epsilon = 1.e-3

    # Remove very small values, rescale the rest
    dd = dict((key, d[key]) for key in d if d[key] > epsilon)
    keys, vals = zip(*[(key, d[key]) for key in dd])
    ss = sum(vals)
    vals = [val / ss * cap for val in vals]
    dd = dict(zip(keys, vals))

    intvals = [int(np.round(x / epsilon)) for x in vals]
    intdist = int(1 / epsilon)
    intdd = dict(zip(keys, intvals))

    s = {}
    t = {}
    taus = []
    sumsofar = 0
    for item in keys:
        s[item] = sumsofar
        t[item] = sumsofar + intdd[item]
        taus.append(t[item] % intdist)
        sumsofar = t[item]

    # print s,t,taus
    taus = sorted(set(taus))
    # print taus

    if intdist not in taus:
        taus.append(intdist)

    placements = {}
    prob = {}

    for i in range(len(taus) - 1):
        x = []
        t_low = taus[i]
        t_up = taus[i + 1]

        diff = t_up - t_low

        for ell in range(int(cap)):
            lower = ell * intdist + t_low
            upper = ell * intdist + t_up
            for item in keys:
                # print lower,upper,' inside ', s[item],t[item], '?',
                if lower >= s[item] and upper <= t[item]:
                    x.append(item)
            #    print ' yes'
            # else: print ' no'
        prob[i] = 1. * diff / intdist
        placements[i] = x

    # print "Placements ",placements,"with prob",prob

    totsum = np.sum(list(prob.values()))
    if not np.allclose(totsum, 1):
        for i in prob:
            prob[i] = 1. * prob[i] / totsum
    # round to 1

    return placements, prob, rv_discrete(values=(list(prob.keys()), list(prob.values())))


def dict_min_val(d):
    min_val = float('inf')
    for key in d:
        if d[key] < min_val:
            min_key = key
            min_val = d[key]
    return min_key, min_val


def path_length(G, path):
    return sum(G[u][v]['weight'] for (u, v) in zip(path[1:], path[:-1]))


def is_simple(path):
    return len(path) == len(set(path))


def generatePaths(G, source, destination, cutoff=20, stretch=1.2):
    node_distances, shortest_paths = single_source_dijkstra(G, destination)
    in_paths = [[source]]
    out_paths = {}
    path_distances = {}
    out_path_set = set()
    excess_budget = stretch * node_distances[source]

    # logging.debug(pp(["Excess budget is ",excess_budget]))

    while in_paths and len(out_paths) < cutoff:
        path = in_paths.pop(0)
        cost_so_far = path_length(G, path)
        logging.debug(pp(['Exploring', path, 'with cost', cost_so_far]))
        last_node = path[-1]
        logging.debug(pp(['Last node is', last_node]))
        shortest_from_here = path + list(reversed(shortest_paths[last_node][:-1]))
        logging.debug(pp(['Shortest path from here is', shortest_from_here]))
        shortest_path_length = path_length(G, shortest_from_here)

        if is_simple(shortest_from_here) and shortest_path_length < excess_budget and tuple(
                shortest_from_here) not in out_path_set:
            logging.debug(pp(['Adding', shortest_from_here, 'with length', shortest_path_length, 'to out_paths']))
            path_id = len(out_paths)
            out_paths[path_id] = shortest_from_here
            path_distances[path_id] = shortest_path_length
            out_path_set.add(tuple(shortest_from_here))

        non_visited = set(G.neighbors(last_node)).difference(path)
        logging.debug(pp(['Non visited neighbors are', non_visited]))
        for new_node in non_visited:
            new_path = path + [new_node]
            new_path_length = path_length(G, new_path)
            if new_path_length + node_distances[new_node] < excess_budget:
                in_paths += [new_path]
    return out_paths, path_distances


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


def Dependencies(demands):
    """
    Generate a dictionary dependencies: key: (node, item), value: a list of (demand, path)
    """
    dependencies = {}
    for d in range(len(demands)):
        item = demands[d].item
        paths = demands[d].routing_info['paths']
        for p in demands[d].routing_info['paths']:
            path = paths[p]
            x = demands[d].query_source
            s = succFun(x, path)
            while s is not None:
                if (x, item) not in dependencies:
                    dependencies[(x, item)] = [(d,p)]
                else:
                    dependencies[(x, item)].append((d,p))
                x = s
                s = succFun(x, path)
    return dependencies


if __name__ == "__main__":
    G = grid_2d_graph(50, 50)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = random()
    source = (0, 0)
    destination = (49, 49)
    paths, distances = generatePaths(G, source, destination, cutoff=100, stretch=1.5)
    print(paths, distances)
