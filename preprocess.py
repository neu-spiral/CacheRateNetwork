import pandas as pd
import pickle
import networkx as nx

capacities_pd = pd.read_excel('data/topology&capacity.xlsx', sheet_name='storage capacity', header=None)
bandwidths_pd = pd.read_excel('data/topology&capacity.xlsx', sheet_name='link capacity', header=None)
traces_pd = pd.read_csv("data/Real trace.csv", header=None)

capacities_val = capacities_pd.values
bandwidths_val = bandwidths_pd.values
traces_val = traces_pd.iloc[:, 1:3].values
capacities = dict(capacities_val)
bandwidths = {}
for i, j, k in bandwidths_val:
    bandwidths[(i,j)] = k

'''find the largest connected subgraph'''
G = nx.Graph()
G.add_nodes_from(capacities.keys())
G.add_edges_from(bandwidths.keys())
number_graph = 0
largest_connect = None
for connect in nx.connected_components(G):
    if len(connect) > number_graph:
        largest_connect = connect
        number_graph = len(connect)

'''generate capacities, bandwidths and traces in the subgraph'''
capacities_connect = {}
for node in largest_connect:
    capacities_connect[node] = capacities[node]

bandwidths_connect = {}
for i, j in bandwidths:
    if i in largest_connect and j in largest_connect:
        bandwidths_connect[(i,j)] = bandwidths[(i,j)]

items, query_nodes = zip(*traces_val)

items_connect = []
query_nodes_connect = []
for i in range(len(query_nodes)):
    if query_nodes[i] in largest_connect:
        query_nodes_connect.append(query_nodes[i])
        items_connect.append(items[i])


traces = {}
for i in range(len(items_connect)):
    item = items_connect[i]
    query_node = query_nodes_connect[i]
    if (item, query_node) in traces:
        traces[(item, query_node)] += 1
    else:
        traces[(item, query_node)] = 1

traces = list(traces.items())
traces.sort(key=lambda x: x[1], reverse=True)
traces_2000 = traces[:5000]

keys, values = zip(*traces_2000)
items, query_nodes = zip(*keys)

items_set = set(items)
item_map = dict(zip(items_set, range(len(items_set))))

traces_2000 = dict(traces_2000)
traces = {}
for (item, query_node) in traces_2000:
    traces[(item_map[item],query_node)] = traces_2000[(item, query_node)]

result = (capacities_connect, bandwidths_connect, traces, list(item_map.values()), set(query_nodes))
fname = 'real_5000'
with open(fname, 'wb') as f:
    pickle.dump(result, f)