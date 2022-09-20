import matplotlib.pyplot as plt
import logging, argparse
import pickle
import numpy as np

topology = ['abilene2', 'geant', 'real1']  #topology used in graph
topology_map = {'abilene2': 'Abilene2', 'geant': 'GEANT', 'real1': 'KS1'}
bandwidth_coefficients = [1, 1.5, 2, 2.5, 3]
algorithm = ['PrimalDual', 'Random1', 'Random2', 'Greedy1', 'Greedy2', 'Alternating']
Dirs = {1: ["OUTPUT1/", "Random1/CacheRoute/", "Random1/RouteCache/", "Greedy1/CacheRoute/", "Greedy1/RouteCache/", "Heuristic1/"],
        2: ["OUTPUT2/", "Random2/CacheRoute/", "Random2/RouteCache/", "Greedy2/CacheRoute/", "Greedy2/RouteCache/", "Heuristic2/"],
        3: ["OUTPUT3/", "Random3/CacheRoute/", "Random3/RouteCache/", "Greedy3/CacheRoute/", "Greedy3/RouteCache/", "Heuristic3/"]}

colors = ['r', 'sandybrown', 'gold', 'darkseagreen', 'c', 'dodgerblue', 'm']
line_styles = ['s-', '*-', 'd--', '^-', 'v-', '.:']

def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def plotJointPlot(x, type):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(12, 4)
    for k in range(len(x)):
        for i in range(len(algorithm)):
            alg = algorithm[i]
            for j in range(len(x[k][alg])):
                if x[k][alg][j]:
                    break
            else:
                j = len(x[k][alg])
            ax[k].plot(bandwidth_coefficients[j:], x[k][alg][j:], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[k].tick_params(labelsize=10)
            ax[k].set_ylabel('Cache Gain $F$', fontsize=15)
            ax[k].set_xlabel('Looseness $\kappa$', fontsize=15)
            ax[k].set_title(topology_map[topology[k]], fontsize=13)
    plt.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(algorithm), fontsize=13)
    plt.show()
    fig.savefig('Figure/sens%d/joint.pdf' % (type), bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot bar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--catalog_size', default=1000, type=int, help='Catalog size')
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--query_nodes', default=4, type=int, help='Number of nodes generating queries')
    parser.add_argument('--demand_size', default=5000, type=int, help='Demand size')
    parser.add_argument('--max_capacity', default=20, type=int, help='Maximum capacity per cache')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--stepsize', default=1000, type=int, help='Stepsize')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj = []
    for k in range(len(topology)):
        obj.append({})
        for alg in algorithm:
            obj[k][alg] = []
    Dir = Dirs[args.bandwidth_type]

    for k in range(len(topology)):
        graph_type = topology[k]
        for bandwidth_coefficient in bandwidth_coefficients:

            fname1 = "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
                graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
                bandwidth_coefficient)

            result = 0
            vio = 1000000
            for stepsize in [100, 500, 1000, 5000, 10000]:
                fname = Dir[0] + fname1 + "_%dstepsize" % (stepsize)
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
                        if (objs[i] >= 0.999 * result and vio_min < vio) or (objs[i] > result and vio_min <= 1.001 * vio):
                            result = objs[i]
                            vio = vio_min

            obj[k][algorithm[0]].append(result)

            for i in range(1, len(algorithm)-1):
                fname = Dir[i] + fname1
                result = readresult(fname)
                result = result[-1]
                obj[k][algorithm[i]].append(result)

            fname = Dir[-1] + fname1
            result = readresult(fname)
            result = result[-1][-1]
            obj[k][algorithm[-1]].append(result)

    plotJointPlot(obj, args.bandwidth_type)