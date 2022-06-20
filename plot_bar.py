import matplotlib.pylab as plt
import argparse, pickle
import numpy as np

topology_map = {'erdos_renyi': 'ER', 'grid_2d': 'grid', 'hypercube': 'HC', 'balanced_tree': 'BT', 'small_world': 'SW'}
topology_small_map = {'geant': 'geant', 'dtelekom': 'dtelekom', 'abilene': 'abilene1', 'abilene2': 'abilene2',
                      'example1': 'ex1', 'example2': 'ex2'}


algorithm = ['PrimalDual', 'Greedy1', 'Greedy2', 'Random1', 'Random2', 'Heuristic']
Dirs = {1: ["OUTPUT10/", "Greedy1/CacheRoute/", "Greedy1/RouteCache/", "Random1/CacheRoute/", "Random1/RouteCache/", "Heuristic1/"],
        2: ["OUTPUT11/", "Greedy2/CacheRoute/", "Greedy2/RouteCache/", "Random2/CacheRoute/", "Random2/RouteCache/", "Heuristic2/"],
        3: ["OUTPUT12/", "Greedy3/CacheRoute/", "Greedy3/RouteCache/", "Random3/CacheRoute/", "Random3/RouteCache/", "Heuristic3/"]}

colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def saveviolation(x, type, bandwidth_coefficient):
    fname = 'Figure/top%d/topologies%f.txt' % (type, bandwidth_coefficient)
    with open(fname, 'w') as f:
        for k, v in x.items():
            f.write(k + ': ' + str(v))
            f.write('\n')


def barplot(x, type, bandwidth_coefficient):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3)
    N = len(topology_map) + len(topology_small_map)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], color=colors[i], label=algorithm[i])
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Cache Gain', fontsize=15)
    ax.set_xlabel('Topology', fontsize=15)

    ax.set_xticks(ind + width*(len(algorithm)-1)/2)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    # plt.ylim(0.5, 1.5)
    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(algorithm), fontsize=13)
    plt.grid(axis='y', linestyle='--')
    plt.show()
    fig.savefig('Figure/top%d/topologies%f.pdf' % (type, bandwidth_coefficient), bbox_extra_artists=(lgd,), bbox_inches = 'tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot bar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    # parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=5, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--stepsize', default=100, type=int, help='Stepsize')

    args = parser.parse_args()

    obj = {}
    for alg in algorithm:
        obj[alg] = {}
        for top in topology_map.values():
            obj[alg][top] = 0
        for top in topology_small_map.values():
            obj[alg][top] = 0

    violation = {}
    for top in topology_map.values():
        obj[top] = 0
    for top in topology_small_map.values():
        obj[top] = 0

    fname1 = "_%ditems_%dnodes_10querynodes_%ddemands_%dcapcity_%fbandwidth_%dstepsize" % (
        args.catalog_size, args.graph_size, args.demand_size, args.max_capacity, args.bandwidth_coefficient, args.stepsize)

    fname2 = "_%ditems_%dnodes_10querynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.catalog_size, args.graph_size, args.demand_size, args.max_capacity, args.bandwidth_coefficient)


    fname3 = "_%ditems_%dnodes_4querynodes_%ddemands_%dcapcity_%fbandwidth_%dstepsize" % (
        args.catalog_size, args.graph_size, args.demand_size, args.max_capacity, args.bandwidth_coefficient, args.stepsize)

    fname4 = "_%ditems_%dnodes_4querynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.catalog_size, args.graph_size, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    Dir = Dirs[args.bandwidth_type]
    for top in topology_map:

        fname = Dir[0] + top + fname1
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
        violation[top] = vio_min

        '''obj'''
        compare = 0
        for i in range(len(objs)):
            if vios[i] == vio_min:
                compare = max(compare, objs[i])
        obj[algorithm[0]][topology_map[top]] = compare / compare

        for j in range(1, len(algorithm)-1):
            fname = Dir[j] + top + fname2
            result = readresult(fname)
            obj[algorithm[j]][topology_map[top]] = result[-1] / compare

        fname = Dir[-1] + top + fname2
        result = readresult(fname)
        obj[algorithm[-1]][topology_map[top]] = result[-1][-1] / compare

    for top in topology_small_map:

        fname = Dir[0] + top + fname3
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
        violation[top] = vio_min

        '''obj'''
        compare = 0
        for i in range(len(objs)):
            if vios[i] == vio_min:
                compare = max(compare, objs[i])
        obj[algorithm[0]][topology_small_map[top]] = compare / compare

        for j in range(1, len(algorithm)-1):
            fname = Dir[j] + top + fname4
            result = readresult(fname)
            obj[algorithm[j]][topology_small_map[top]] = result[-1] / compare

        fname = Dir[-1] + top + fname4
        result = readresult(fname)
        obj[algorithm[-1]][topology_small_map[top]] = result[-1][-1] / compare

    saveviolation(violation, args.bandwidth_type, args.bandwidth_coefficient)
    barplot(obj, args.bandwidth_type, args.bandwidth_coefficient)

