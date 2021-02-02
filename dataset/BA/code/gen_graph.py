import networkx as nx
import numpy as np

np.random.seed(42)

# -------------
# parameters
# --------------

path = '../data/graph/'

N = 1000 # number of nodes
m = 10 # random graph probability to connect

#print(2 * np.log(N)/ N)

for graph_id in range(5):
    file_name = path + 'BA_N%s_m%s_g%s.edgelist' % (N, m, graph_id)

    # Network Definition
    g = nx.barabasi_albert_graph(N, m)

    nx.write_edgelist(g, file_name, data=False)
