import matplotlib.pyplot as plt
import pickle
import numpy as np

fname = 'real_5000'
with open(fname, 'rb') as f:
    (capacities_real, bandwidths_real, traces_real, catalog_real, query_nodes_real) = pickle.load(f)

nodes = list(capacities_real.keys())
nodes = set(nodes)
number_map = dict(zip(nodes, range(len(nodes))))
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)
ys = []
xs = []
sizes = []
for item, query_node in traces_real:
    ys.append(item)
    xs.append(number_map[query_node])
    sizes.append(traces_real[(item, query_node)]/10)
ax.scatter(xs, ys, s=sizes)
ax.tick_params(labelsize=10)
ax.set_ylabel('Item', fontsize=15)
ax.set_xlabel('Query Node', fontsize=15)
plt.show()
fig.savefig('Figure/trace_' + fname + '.pdf', bbox_inches='tight')
