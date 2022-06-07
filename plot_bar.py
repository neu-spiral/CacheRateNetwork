import matplotlib.pylab as plt
import argparse, pickle
import numpy as np

topology = ['erdos_renyi', 'grid_2d', 'hypercube', 'star']
topology_short = ['ER', 'grid', 'HC', 'star']
# topology = ['erdos_renyi', 'balanced_tree', 'hypercube', 'star', 'geant', 'abilene', 'dtelekom', 'grid_2d', 'small_world']
# topology_short = ['ER', 'BT', 'HC', 'star', 'geant', 'abilene', 'dtelekom', 'grid', 'SW']
algorithm = ['PrimalDual', 'Greedy1', 'Greedy2', 'Random1', 'Random2', 'Heuristic']
Dir = ["OUTPUT6/", "Greedy/CacheRoute/", "Greedy/RouteCache/", "Random/CacheRoute/", "Random/RouteCache/", "Heuristic/"]
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3)
    N = len(topology)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Cache Gain', fontsize=15)

    ax.set_xticks(ind + width*(len(algorithm)-1)/2)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    # plt.ylim(0.5, 1.5)
    lgd = fig.legend(labels = algorithm, loc='upper center', ncol=len(algorithm), fontsize=15)
    plt.show()
    # fig.savefig('Figure/topology2_beta.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=5, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--stepsize', default=50, type=int, help='Stepsize')

    args = parser.parse_args()

    obj = {}
    for alg in algorithm:
        obj[alg] = {}
        for top in topology_short:
            obj[alg][top] = 0

    fname1 = "_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth_%dstepsize" % (
        args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
        args.bandwidth_coefficient, args.stepsize)

    fname2 = "_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
        args.bandwidth_coefficient)

    for i in range(len(topology)):

        fname = Dir[0] + topology[i] + fname1
        result = readresult(fname)
        obj[algorithm[0]][topology_short[i]] = result[-1][-1]

        for j in range(1, len(algorithm)-1):
            fname = Dir[j] + topology[i] + fname2
            result = readresult(fname)
            obj[algorithm[j]][topology_short[i]] = result[-1]

        fname = Dir[len(algorithm)-1] + topology[i] + fname2
        result = readresult(fname)
        obj[algorithm[len(algorithm)-1]][topology_short[i]] = result[-1][-1]

    barplot(obj)

