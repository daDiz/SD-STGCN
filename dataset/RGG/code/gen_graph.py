# --------------------------------------------
# generate random geometric graph in 2D
# --------------------------------------


import networkx as nx
import numpy as np

np.random.seed(42)

# -------------
# parameters
# --------------

path = '../data/graph/'

N = 1000 # number of nodes
r = 0.08 # radius, two nodes are joined by an edge if the distance between them is at most radius


for graph_id in range(5):
    file_name = path + 'RGG_N%s_r%s_g%s.edgelist' % (N, r, graph_id)

    # Network Definition
    g = nx.random_geometric_graph(N, r, dim=2)

    nx.write_edgelist(g, file_name, data=False)
