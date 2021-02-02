import networkx as nx
import numpy as np
import sys

np.random.seed(42)

# -------------
# parameters
# --------------

path = '../data/graph/'

N = int(sys.argv[1]) # 1000 # number of nodes
p = 0.02 # random graph probability to connect


for graph_id in range(5):
    file_name = path + 'ER_N%s_p%s_g%s.edgelist' % (N, p, graph_id)

    # Network Definition
    g = nx.erdos_renyi_graph(N, p)

    nx.write_edgelist(g, file_name, data=False)
