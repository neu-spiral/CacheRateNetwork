import matplotlib.pyplot as plt
import logging, argparse
import pickle
import numpy as np

bandwidth_coefficients = [1, 1.5, 2, 2.5, 3]
algorithm = ['PrimalDual', 'Greedy1', 'Greedy2', 'Random1', 'Random2', 'Heuristic']
Dirs = {1: ["OUTPUT10/", "Greedy1/CacheRoute/", "Greedy1/RouteCache/", "Random1/CacheRoute/", "Random1/RouteCache/", "Heuristic1/"],
        2: ["OUTPUT11/", "Greedy2/CacheRoute/", "Greedy2/RouteCache/", "Random2/CacheRoute/", "Random2/RouteCache/", "Heuristic2/"],
        3: ["OUTPUT12/", "Greedy3/CacheRoute/", "Greedy3/RouteCache/", "Random3/CacheRoute/", "Random3/RouteCache/", "Heuristic3/"]}

colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
line_styles = ['s-', '*-', 'd--', '^-', 'v-', '.:']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def plotSensitivity(x, type, graph):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    for i in range(len(algorithm)):
        alg = algorithm[i]
        for j in range(len(x[alg])):
            if x[alg][j]:
                break
        else:
            j = len(x[alg])
        ax.plot(bandwidth_coefficients[j:], x[alg][j:], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Cache Gain', fontsize=15)
    ax.set_xlabel('Looseness', fontsize=15)

    lgd = fig.legend(fontsize=13, loc='upper left', bbox_to_anchor=(0.95, 0.9))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()
    fig.savefig('Figure/sens%d/%s.pdf' % (type, graph),  bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot bar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--stepsize', default=100, type=int, help='Stepsize')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj = {}
    for alg in algorithm:
        obj[alg] = []
    Dir = Dirs[args.bandwidth_type]

    for bandwidth_coefficient in bandwidth_coefficients:
        fname1 = "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth_%dstepsize" % (
            args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
            bandwidth_coefficient, args.stepsize)
        fname2 = "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
            args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
            bandwidth_coefficient)

        fname = Dir[0] + fname1
        result = readresult(fname)

        '''calculate violation'''
        SumFlows = []
        NumNonzeroFlows = []
        iterations, Xs, Rs, overflows, Duals, lagrangians, objs = zip(*result)
        for overflow in overflows:
            ActiveFlow = []
            Flow = []
            for e in overflow:
                if overflow[e] > 0:  # violated flow
                    ActiveFlow.append(overflow[e])
                if overflow[e] > -1:  # non zero flow
                    Flow.append(overflow[e])
            if ActiveFlow:
                SumFlows.append(sum(ActiveFlow))
            else:
                SumFlows.append(0)
            if Flow:
                NumNonzeroFlows.append(len(Flow))
            else:
                NumNonzeroFlows.append(0)
        vios = np.array(SumFlows) / np.array(NumNonzeroFlows)
        vio_min = min(vios)

        '''obj'''
        result = 0
        for i in range(len(objs)):
            if vios[i] == vio_min:
                result = max(result, objs[i])
        obj[algorithm[0]].append(result)

        for i in range(1, len(algorithm)-1):
            fname = Dir[i] + fname2
            result = readresult(fname)
            result = result[-1]
            obj[algorithm[i]].append(result)

        fname = Dir[-1] + fname2
        result = readresult(fname)
        result = result[-1][-1]
        obj[algorithm[-1]].append(result)

    plotSensitivity(obj, args.bandwidth_type, args.graph_type)