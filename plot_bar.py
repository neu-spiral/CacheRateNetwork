import matplotlib.pylab as plt
import argparse, pickle
import numpy as np

topology_map = {'erdos_renyi': 'ER', 'grid_2d': 'grid', 'hypercube': 'HC', 'balanced_tree': 'BT', 'small_world': 'SW'}
topology_small_map = {'geant': 'GEANT', 'dtelekom': 'dtelekom', 'abilene1': 'abilene1', 'Abilene2': 'Abilene2',
                      'example1': 'example1', 'example2': 'example2', 'real1': 'KS1', 'real2': 'KS2'}


algorithm = ['PrimalDual', 'Random1', 'Random2', 'Greedy1', 'Greedy2', 'Alternating']
Dirs = {1: ["OUTPUT1/", "Random1/CacheRoute/", "Random1/RouteCache/", "Greedy1/CacheRoute/", "Greedy1/RouteCache/", "Heuristic1/"],
        2: ["OUTPUT2/", "Random2/CacheRoute/", "Random2/RouteCache/", "Greedy2/CacheRoute/", "Greedy2/RouteCache/", "Heuristic2/"],
        3: ["OUTPUT3/", "Random3/CacheRoute/", "Random3/RouteCache/", "Greedy3/CacheRoute/", "Greedy3/RouteCache/", "Heuristic3/"]}

colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
hatches = ['/', '\\\\', '|', 'o', '--', '', '////',  'x', '+', '.', '\\']


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


def barplot(x1, x2, x3, type, bandwidth_coefficient):
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(18, 8)
    N = len(topology_map) + len(topology_small_map)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x1[algorithm[i]].values()
        ax[0].bar(ind+i*width, y_ax, width=width, hatch=hatches[i], color=colors[i], label=algorithm[i], edgecolor='k', linewidth=1.5)
    ax[0].tick_params(labelsize=10)
    ax[0].set_ylabel('Norm. Cache Gain $F/F_{PD}$', fontsize=15)
    ax[0].set_xlabel('Topology', fontsize=15)
    ax[0].set_xticks(ind + width*(len(algorithm)-1)/2)
    ax[0].set_xticklabels(x1[algorithm[i]].keys(), fontsize=13)
    ax[0].grid(axis='y', linestyle='--')

    for i in range(len(algorithm)):
        y_ax = x2[algorithm[i]].values()
        ax[1].bar(ind+i*width, y_ax, width=width, hatch=hatches[i], color=colors[i], label=algorithm[i], edgecolor='k', linewidth=1.5, log=True)
    ax[1].tick_params(labelsize=10)
    ax[1].set_ylabel('Infeasibility $InF$', fontsize=15)
    ax[1].set_xlabel('Topology', fontsize=15)
    ax[1].set_xticks(ind + width*(len(algorithm)-1)/2)
    ax[1].set_xticklabels(x2[algorithm[i]].keys(), fontsize=13)
    ax[1].grid(axis='y', linestyle='--')

    for i in range(len(algorithm)):
        y_ax = x3[algorithm[i]].values()
        ax[2].bar(ind+i*width, y_ax, width=width, hatch=hatches[i], color=colors[i], label=algorithm[i], edgecolor='k', linewidth=1.5, log=True)
    ax[2].tick_params(labelsize=10)
    ax[2].set_ylabel('Time', fontsize=15)
    ax[2].set_xlabel('Topology', fontsize=15)
    ax[2].set_xticks(ind + width*(len(algorithm)-1)/2)
    ax[2].set_xticklabels(x3[algorithm[i]].keys(), fontsize=13)
    ax[2].grid(axis='y', linestyle='--')

    # plt.ylim(0.5, 1.5)
    plt.tight_layout()
    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(algorithm), fontsize=13)
    plt.show()
    fig.savefig('Figure/top%d/large_topologies%f.pdf' % (type, bandwidth_coefficient), bbox_extra_artists=(lgd,), bbox_inches = 'tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot bar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--catalog_size', default=1000, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    # parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=5000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=20, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--stepsize', default=100, type=int, help='Stepsize')

    args = parser.parse_args()

    obj = {}
    violation = {}
    run_time = {}
    for alg in algorithm:
        obj[alg] = {}
        violation[alg] = {}
        run_time[alg] = {}

        for top in topology_map.values():
            obj[alg][top] = 0
            violation[alg][top] = 0
            run_time[alg][top] = 0

        for top in topology_small_map.values():
            obj[alg][top] = 0
            violation[alg][top] = 0
            run_time[alg][top] = 0

    cmp = {}
    for top in topology_map.keys():
        cmp[top] = 0
    for top in topology_small_map.keys():
        cmp[top] = 0

    fname1 = "_%ditems_%dnodes_10querynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.catalog_size, args.graph_size, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    fname2 = "_%ditems_%dnodes_4querynodes_%ddemands_%dcapcity_%fbandwidth" % (
        args.catalog_size, args.graph_size, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    def collectData(topology_map, fname1):
        Dir = Dirs[args.bandwidth_type]
        for top in topology_map:

            compare = 0
            vio = 1000000
            for stepsize in [100, 500, 1000, 5000, 10000]:
                fname = Dir[0] + top + fname1 + "_%dstepsize" % (stepsize)
                results = readresult(fname)

                '''calculate violation'''
                SumFlows = []
                NumNonzeroFlows = []
                iterations, durations, Xs, Rs, overflows, Duals, lagrangians, objs = zip(*results)
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
                for i in range(len(objs)):
                    if vios[i] == vio_min:
                        if (objs[i] >= 0.999 * compare and vio_min < vio) or (objs[i] > compare and vio_min <= 1.001 * vio):
                            compare = objs[i]
                            vio = vio_min
                            duration = sum(durations)

            obj[algorithm[0]][topology_map[top]] = compare / compare
            violation[algorithm[0]][topology_map[top]] = vio
            run_time[algorithm[0]][topology_map[top]] = duration

            cmp[top] = compare

            for j in range(1, len(algorithm)-1):
                fname = Dir[j] + top + fname1
                result = readresult(fname)
                obj[algorithm[j]][topology_map[top]] = result[-1] / compare
                if result[-1] == 0:
                    violation[algorithm[j]][topology_map[top]] = 1000000
                else:
                    overflow = result[-2]
                    ActiveFlow = []
                    Flow = []
                    # print(overflow, fname)
                    for e in overflow:
                        if overflow[e] > 0:  # violated flow
                            ActiveFlow.append(overflow[e])
                        if overflow[e] > -1:  # non zero flow
                            Flow.append(overflow[e])
                    if ActiveFlow:
                        SumFlows = sum(ActiveFlow)
                    else:
                        SumFlows = 0
                    if Flow:
                        NumNonzeroFlows = len(Flow)
                    else:
                        NumNonzeroFlows = 0
                    vio = SumFlows / NumNonzeroFlows
                    violation[algorithm[j]][topology_map[top]] = vio
                run_time[algorithm[j]][topology_map[top]] = result[0]

            fname = Dir[-1] + top + fname1
            result = readresult(fname)
            obj[algorithm[-1]][topology_map[top]] = result[-1][-1] / compare
            if result[-1][-1] == 0:
                violation[algorithm[-1]][topology_map[top]] = 1000000
            else:
                overflow = result[-1][-2]
                ActiveFlow = []
                Flow = []
                for e in overflow:
                    if overflow[e] > 0:  # violated flow
                        ActiveFlow.append(overflow[e])
                    if overflow[e] > -1:  # non zero flow
                        Flow.append(overflow[e])
                if ActiveFlow:
                    SumFlows = sum(ActiveFlow)
                else:
                    SumFlows = 0
                if Flow:
                    NumNonzeroFlows = len(Flow)
                else:
                    NumNonzeroFlows = 0
                vio = SumFlows / NumNonzeroFlows
                violation[algorithm[-1]][topology_map[top]] = vio
            run_time[algorithm[-1]][topology_map[top]] = sum([res[1] for res in result])


    collectData(topology_map, fname1)
    collectData(topology_small_map, fname2)

    saveviolation(cmp, args.bandwidth_type, args.bandwidth_coefficient)
    barplot(obj, violation, run_time, args.bandwidth_type, args.bandwidth_coefficient)

