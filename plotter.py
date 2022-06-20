import matplotlib.pyplot as plt
import logging, argparse
import pickle
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot convergence',
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
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--stepsize', default=50, type=int, help='Stepsize')

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    # if args.bandwidth_type == 1:
    #     dir = "OUTPUT6/"
    # else:
    #     dir = "OUTPUT%d/" % (args.bandwidth_type+6)
    dir = "OUTPUT%d/" % (args.bandwidth_type + 9)
    fname = dir + "%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth_%dstepsize" % (
        args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity,
        args.bandwidth_coefficient, args.stepsize)

    with open(fname, 'rb') as f:
        results = pickle.load(f)

    iterations, Xs, Rs, overflows, Duals, lagrangians, objs = zip(*results)
    SumDuals = []
    for Dual in Duals:
        SumDual = sum(Dual.values())
        SumDuals.append(SumDual)

    SumFlows = []
    MaxFlows = []
    NumActiveFlows = []
    NumNonzeroFlows = []
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
            MaxFlows.append(max(ActiveFlow))
            NumActiveFlows.append(len(ActiveFlow))
        else:
            SumFlows.append(0)
            MaxFlows.append(0)
            NumActiveFlows.append(0)

        if Flow:
            NumNonzeroFlows.append(len(Flow))
        else:
            NumNonzeroFlows.append(0)

    # fig, ax = plt.subplots(nrows=4, ncols=1)
    #
    # ax[0].plot(iterations, SumDuals, label='SumDual')
    # ax[0].legend()
    # ax[1].plot(iterations, lagrangians, label='L')
    # ax[1].plot(iterations, objs, label='Obj')
    # ax[1].legend()
    # ax[2].plot(iterations, SumFlows, label='SumOverflow')
    # ax[2].plot(iterations, MaxFlows, label='MaxOverflow')
    # if min(SumFlows) < 1:
    #     ax[2].set_ylim([-0.05, 1])
    # ax[2].legend()
    # ax[3].plot(iterations, NumActiveFlows, label='NumActiveflow')
    # ax[3].plot(iterations, NumNonzeroFlows, label='NumNonzeroFlows')
    #
    # ax[3].set_title(str(len(overflow)))
    # ax[3].legend()

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(5, 7)
    ax[0].plot(iterations, lagrangians, label='L')
    ax[0].plot(iterations, objs, '-.', label='Obj')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=13)
    violations = np.array(SumFlows) / np.array(NumNonzeroFlows)
    violations_max = np.array(MaxFlows) / np.array(NumNonzeroFlows)
    ax[1].plot(iterations, violations, label='SumOverflow')
    ax[1].plot(iterations, violations_max, '-.', label='MaxOverflow')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=13)

    plt.tick_params(labelsize=10)
    ax[0].set_ylabel('Cache Gain', fontsize=15)
    ax[0].set_xlabel('Iterations', fontsize=15)
    ax[1].set_ylabel('Overflow', fontsize=15)
    ax[1].set_xlabel('Iterations', fontsize=15)
    plt.tight_layout()
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.show()
    logging.info('Plot ' + fname)

    fig.savefig(fname + '.pdf', bbox_inches = 'tight')
