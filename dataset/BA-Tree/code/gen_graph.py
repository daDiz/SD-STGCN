# ----------------------
# generate BA-tree
# ---------------------

import networkx as nx
import numpy as np

np.random.seed(42)

# -------------
# parameters
# --------------

path = '../data/graph/'

N = 1000 # number of nodes
m = 1 # number of edges to attach from a new node existing nodes
        # ba graphs become ba trees when m = 1


for graph_id in range(5):
    file_name = path + 'BA-Tree_N%s_m%s_g%s.edgelist' % (N, m, graph_id)

    # Network Definition
    g = nx.barabasi_albert_graph(N, m)

    nx.write_edgelist(g, file_name, data=False)
