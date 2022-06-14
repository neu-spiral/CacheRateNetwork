import logging, argparse
import networkx
import random
import numpy as np
from helpers import pp, generatePaths, succFun, path_length, Dependencies

import topologies
import pickle
import os


class Demand:
    """ A demand object. Contains the item requested, any routing information relevant to this demand, and the
        rate with which requests are generated. Tallies count various metrics.

        Attributes:
        item: the id of the item requested
        routing_info: e.g., a path, a list of paths, the source node, etc., depending on the routing policy
        rate: the rate with which this request is generated
           """

    def __init__(self, item, query_source, rate, routing_info=None):
        """ Initialize a new request.
        """
        self.item = item
        self.query_source = query_source
        self.rate = rate
        self.routing_info = routing_info

    def __str__(self):
        return Demand.__repr__(self)

    def __repr__(self):
        return 'Demand(' + ','.join(map(str, [self.item, self.query_source, self.rate])) + ')'


class Problem:
    def __init__(self, graph, capacities, bandwidths, demands, weights):
        self.graph = graph
        self.capacities = capacities
        self.bandwidths = bandwidths
        self.demands = demands
        self.weights = weights

    def pickle_cls(self, fname):
        f = open(fname, 'wb')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def unpickle_cls(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)


def main():
    # logging.basicConfig(filename='execution.log', filemode='w', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Simulate a Network of Caches',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('inputfile',help = 'Training data. This should be a tab separated file of the form: index _tab_ features _tab_ output , where index is a number, features is a json string storing the features, and output is a json string storing output (binary) variables. See data/LR-example.txt for an example.')
    parser.add_argument('outputfile', help='Output file')
    parser.add_argument('--max_capacity', default=5, type=int, help='Maximum capacity per cache')
    parser.add_argument('--min_capacity', default=1, type=int, help='Minimum capacity per cache')
    parser.add_argument('--bandwidth_coefficient', default=1, type=float,
                        help='Coefficient of bandwidth for max flow, this coefficient should be between (1, max_paths)')
    parser.add_argument('--bandwidth_type', default=1, type=int,
                        help='Type of generating bandwidth: 1. no cache, 2. uniform cache, 3. random integer cache')
    parser.add_argument('--max_weight', default=100.0, type=float, help='Maximum edge weight')
    parser.add_argument('--min_weight', default=1.0, type=float, help='Minimum edge weight')
    parser.add_argument('--rate', default=1.0, type=float, help='Average rate per demand')
    parser.add_argument('--max_paths', default=5, type=int, help='Maximum number of paths per demand')
    parser.add_argument('--path_stretch', default=4.0, type=float, help='Allowed stretch from shortest path')
    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    #   parser.add_argument('--sources_per_item',default=1,type=int, help='Number of designated sources per catalog item')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--demand_distribution', default="powerlaw", type=str, help='Demand distribution',
                        choices=['powerlaw', 'uniform'])
    parser.add_argument('--powerlaw_exp', default=1.2, type=float,
                        help='Power law exponent, used in demand distribution')
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'example1', 'example2'])
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--graph_degree', default=3, type=int,
                        help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
    parser.add_argument('--graph_p', default=0.10, type=int, help='Probability, used in erdos_renyi, watts_strogatz')
    parser.add_argument('--random_seed', default=1234567890, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    #   parser.add_argument('--cache_keyword_parameters',default='{}',type=str,help='Networked Cache additional constructor parameters')


    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2213)

    def graphGenerator():
        if args.graph_type == "erdos_renyi":
            return networkx.erdos_renyi_graph(args.graph_size, args.graph_p)
        if args.graph_type == "balanced_tree":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(args.graph_degree)))
            return networkx.balanced_tree(args.graph_degree, ndim)
        if args.graph_type == "cicular_ladder":
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.circular_ladder_graph(ndim)
        if args.graph_type == "cycle":
            return networkx.cycle_graph(args.graph_size)
        if args.graph_type == 'grid_2d':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim, ndim)
        if args.graph_type == 'lollipop':
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.lollipop_graph(ndim, ndim)
        if args.graph_type == 'expander':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.margulis_gabber_galil_graph(ndim)
        if args.graph_type == "hypercube":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(2.0)))
            return networkx.hypercube_graph(ndim)
        if args.graph_type == "star":
            ndim = args.graph_size - 1
            return networkx.star_graph(ndim)
        if args.graph_type == 'barabasi_albert':
            return networkx.barabasi_albert_graph(args.graph_size, args.graph_degree)
        if args.graph_type == 'watts_strogatz':
            return networkx.connected_watts_strogatz_graph(args.graph_size, args.graph_degree, args.graph_p)
        if args.graph_type == 'regular':
            return networkx.random_regular_graph(args.graph_degree, args.graph_size)
        if args.graph_type == 'powerlaw_tree':
            return networkx.random_powerlaw_tree(args.graph_size)
        if args.graph_type == 'small_world':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.navigable_small_world_graph(ndim)
        if args.graph_type == 'geant':
            return topologies.GEANT()
        if args.graph_type == 'dtelekom':
            return topologies.Dtelekom()
        if args.graph_type == 'abilene':
            return topologies.Abilene()
        if args.graph_type == 'servicenetwork':
            return topologies.ServiceNetwork()
        if args.graph_type == 'example1' or args.graph_type == 'example2':
            return topologies.example1()

    construct_stats = {}

    logging.info('Generating graph and weights...')
    temp_graph = graphGenerator()
    # networkx.draw(temp_graph)
    # plt.draw()
    logging.debug('nodes: ' + str(temp_graph.nodes()))
    logging.debug('edges: ' + str(temp_graph.edges()))
    G = networkx.DiGraph()

    number_map = dict(zip(temp_graph.nodes(), range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())
    weights = {}
    if args.graph_type == 'example1' or args.graph_type == 'example2':
        example1_weights = topologies.example1_weights(1000)
        for (x, y) in temp_graph.edges():
            xx = number_map[x]
            yy = number_map[y]
            G.add_edges_from(((xx, yy), (yy, xx)))
            weights[(xx, yy)] = example1_weights[(x, y)]
            weights[(yy, xx)] = weights[(xx, yy)]
            G[xx][yy]['weight'] = weights[(xx, yy)]
            G[yy][xx]['weight'] = weights[(yy, xx)]
    else:
        for (x, y) in temp_graph.edges():
            xx = number_map[x]
            yy = number_map[y]
            G.add_edges_from(((xx, yy), (yy, xx)))
            weights[(xx, yy)] = random.uniform(args.min_weight, args.max_weight)
            weights[(yy, xx)] = weights[(xx, yy)]
            G[xx][yy]['weight'] = weights[(xx, yy)]
            G[yy][xx]['weight'] = weights[(yy, xx)]
    graph_size = G.number_of_nodes()
    edge_size = G.number_of_edges()
    logging.info('...done. Created graph with %d nodes and %d edges' % (graph_size, edge_size))
    logging.debug('G is:' + str(G.nodes()) + str(G.edges()))
    construct_stats['graph_size'] = graph_size
    construct_stats['edge_size'] = edge_size

    logging.info('Generating item sources...')
    if args.graph_type == 'example1' or args.graph_type == 'example2':
        item_sources = {0: [0], 1: [1]}
    else:
        item_sources = dict((item, [list(G.nodes())[source]]) for item, source in
                            zip(range(args.catalog_size), np.random.choice(range(graph_size), args.catalog_size)))
    logging.info('...done. Generated %d sources' % len(item_sources))
    logging.debug('Generated sources:')
    for item in item_sources:
        logging.debug(pp([item, ':', item_sources[item]]))

    construct_stats['sources'] = len(item_sources)

    logging.info('Generating query node list...')
    if args.graph_type == 'example1' or args.graph_type == 'example2':
        query_node_list = [5, 6]
    else:
        query_node_list = [list(G.nodes())[i] for i in random.sample(range(graph_size), args.query_nodes)]
    logging.info('...done. Generated %d query nodes.' % len(query_node_list))

    construct_stats['query_nodes'] = len(query_node_list)

    logging.info('Generating demands...')
    if args.graph_type == 'example1' or args.graph_type == 'example2':
        example1_demands = topologies.example1_demands()
        demands = []
        rate = args.rate
        for (item, x) in example1_demands:
            paths = example1_demands[(item, x)]
            distances = {}
            for path_id in paths:
                path = paths[path_id]
                for i in range(len(path)):
                    path[i] = number_map[path[i]]
                distances[path_id] = path_length(G, paths[path_id])

            x = number_map[x]
            logging.debug(pp(['Generated ', len(paths), 'paths for new demand', (item, x, rate)]))
            routing_info = {'paths': paths, 'distances': distances}
            new_demand = Demand(item, x, rate, routing_info=routing_info)
            demands.append(new_demand)
            logging.debug(pp(['Generated demand', new_demand]))

    else:
        if args.demand_distribution == 'powerlaw':
            factor = lambda i: (1.0 + i) ** (-args.powerlaw_exp)
        else:
            factor = lambda i: 1.0
        # normalizing constant so that average rate per demand is args.rate
        constant = args.rate * args.demand_size / sum((factor(i) for i in range(args.demand_size)))
        all_demands = [(x, item) for x in query_node_list for item in range(args.catalog_size)]
        if args.demand_size > len(all_demands):
            demand_pairs = [random.choice(all_demands) for i in range(args.demand_size)]
        else:
            demand_pairs = random.sample(all_demands, args.demand_size)

        counter = 0
        demands = []
        for x, item in demand_pairs:
            rate = constant * factor(counter)
            paths, distances = generatePaths(G, x, item_sources[item][0], cutoff=args.max_paths, stretch=args.path_stretch)
            logging.debug(pp(['Generated ', len(paths), 'paths for new demand', (item, x, rate)]))
            if len(paths) == 0:
                logging.warning(pp(['No paths exist for new demand', (item, x, rate), 'with target', item_sources[item][0],
                                    ', this demand will be dropped']))
                continue
            routing_info = {'paths': paths, 'distances': distances}
            new_demand = Demand(item, x, rate, routing_info=routing_info)
            demands.append(new_demand)
            logging.debug(pp(['Generated demand', new_demand]))
            counter += 1
    logging.info('...done. Generated %d demands' % len(demands))

    construct_stats['demands'] = len(demands)

    logging.info('Generating capacities...')
    if args.graph_type == 'example1' or args.graph_type == 'example2':
        example1_capacities = topologies.example1_capacities()
        capacities = dict((x, 0) for x in G.nodes())
        for node in example1_capacities:
            capacities[number_map[node]] = example1_capacities[node]
    else:
        capacities = dict((x, random.randint(args.min_capacity, args.max_capacity)) for x in G.nodes())
    logging.info('...done. Generated %d caches' % len(capacities))
    logging.debug('Generated capacities:')
    for key in capacities:
        logging.debug(pp([key, ':', capacities[key]]))

    logging.info('Generating bandwidth...')
    # bandwidths = dict((x, random.uniform(args.min_bandwidth, args.max_bandwidth)) for x in G.edges())
    bandwidths = {}
    if args.graph_type == 'example1' or args.graph_type == 'example2':
        if args.graph_type == 'example1':
            example_bandwidths = topologies.example1_bandwidths(args.rate, 0.1, len(demands) * args.rate)
        elif args.graph_type == 'example2':
            example_bandwidths = topologies.example2_bandwidths(args.rate, 0.1, len(demands) * args.rate)
        for (x, y) in temp_graph.edges():
            xx = number_map[x]
            yy = number_map[y]
            bandwidths[(xx, yy)] = example_bandwidths[(x, y)]
            bandwidths[(yy, xx)] = bandwidths[(xx, yy)]
    else:
        '''Random Cache'''
        X = {}
        dependencies = Dependencies(demands)
        for (v, i) in dependencies:
            if args.bandwidth_type == 1:
                if v in X:
                    X[v][i] = 0
                else:
                    X[v] = {i: 0}
            else:
                if v in X:
                    X[v][i] = 1
                else:
                    X[v] = {i: 1}

        for v in X:
            item_total = len(X[v])
            if args.bandwidth_type == 2:
                cache_average = capacities[v] / item_total
                for i in X[v]:
                    X[v][i] = min(cache_average, 1)
            elif args.bandwidth_type == 3:
                sampled_items = random.sample(list(X.keys()), min(capacities[v], item_total))
                for i in X[v]:
                    if i in sampled_items:
                        X[v][i] = 1
                    else:
                        X[v][i] = 0

        for d in demands:
            item = d.item
            rate = d.rate
            paths = d.routing_info['paths']
            max_paths = len(paths)

            for path_id in paths:
                path = paths[path_id]
                x = d.query_source
                s = succFun(x, path)
                prodsofar = 1
                while s is not None:
                    prodsofar *= (1 - X[x][item])
                    if (s, x) in bandwidths:
                        bandwidths[(s, x)] += args.bandwidth_coefficient * rate / max_paths * prodsofar
                    else:
                        bandwidths[(s, x)] = args.bandwidth_coefficient * rate / max_paths * prodsofar
                    x = s
                    s = succFun(x, path)
        for e in bandwidths:
            if bandwidths[e] == 0:
                bandwidths[e] = 0.001

    logging.info('...done. Generated %d bandwidths' % len(bandwidths))
    logging.debug('Generated bandwidth:')
    for key in bandwidths:
        logging.debug(pp([key, ':', bandwidths[key]]))

    logging.info('Building CacheRouteNetwork')

    ''' pack the graph, capacity for each node, attributes of each demands(requests), bandwidth for each edge '''
    pr = Problem(G, capacities, bandwidths, demands, weights)
    dir = "INPUT%d/" % (args.bandwidth_type)
    if not os.path.exists(dir):
        os.mkdir(dir)
    out = dir + args.outputfile + "_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%dcapcity_%fbandwidth" % (
    args.graph_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.max_capacity, args.bandwidth_coefficient)

    pr.pickle_cls(out)  # can only pickle functions defined at the top level of a module
    logging.info('Save data to ' + out)
    logging.info('...done')


if __name__ == "__main__":
    main()