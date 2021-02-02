import time
import numpy as np
import pickle
import networkx as nx
from dmp_dgl_utils import *
from metric_utils import *
from scipy.sparse.linalg import eigs
from scipy import sparse
import sys

np.random.seed(42)

# get the snapshot at time step
# step 0 is the initial state with the source
# step 1 is the first iteration
def get_one_snapshot(iterations, step):
    if step > len(iterations):
        raise Exception('step must be within len(iterations)')

    g = iterations[0]['status'].copy()
    N = len(g)
    for i in range(1, step+1):
        s = iterations[i]['status']
        if len(s) > 0:
            for k in s:
                g[k] = s[k]

    return [g[k] for k in range(N)]



gid = int(sys.argv[1])
T = int(sys.argv[2])

node_start = int(sys.argv[3])
node_end = int(sys.argv[4])

gt = sys.argv[5]

R0 = 2.5
gamma = 0.4

train_pct = 0.8
val_pct = 0.1

if gt == 'ER':
    # ---------
    # ER
    # -----------
    g = nx.read_edgelist('../../dataset/ER/data/graph/ER_N1000_p0.02_g%s.edgelist' % (gid), nodetype=int)
    x, y = pickle.load(open('../../dataset/ER/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_p0.02_g%s_entire.pickle' % (gid), 'rb'))
elif gt == 'BA':
    # ----------
    # BA
    # ------------
    g = nx.read_edgelist('../../dataset/BA/data/graph/BA_N1000_m10_g%s.edgelist' % (gid), nodetype=int)
    x, y =\
    pickle.load(open('../../dataset/BA/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_m10_g%s_entire.pickle' % (gid), 'rb'))
elif gt == 'BA-Tree':
    # ----------
    # BA-Tree
    # ------------
    g = nx.read_edgelist('../../dataset/BA-Tree/data/graph/BA-Tree_N1000_m1_g%s.edgelist' % (gid), nodetype=int)
    x, y =\
    pickle.load(open('../../dataset/BA-Tree/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_m1_g%s_entire.pickle' % (gid), 'rb'))
elif gt == 'RGG':
    # ----------
    # RGG
    # ------------
    g = nx.read_edgelist('../../dataset/RGG/data/graph/RGG_N1000_r0.08_g%s.edgelist' % (gid), nodetype=int)
    x, y =\
    pickle.load(open('../../dataset/RGG/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_r0.08_g%s_entire.pickle' % (gid), 'rb'))


n_tot = len(x)
n_train = int(n_tot * train_pct)
n_val = int(n_tot * val_pct)

x_test = np.array(x)[n_train+n_val:]
y_test = np.array(y)[n_train+n_val:]


A = nx.adjacency_matrix(g).todense().astype(float)
lambda_max = eigs(A, k=1, which='LR')[0][0].real
beta = round(R0 * gamma / lambda_max, 3)


dmp = DMP(g, lamb=beta, mu=gamma)

out_file = 'dmp_pred_%s-%s-%s.pickle' % (T, node_start, node_end)

y_pred = []
s0 = time.time()
for sample_ind in range(len(x_test)):
    snapshot = get_one_snapshot(x_test[sample_ind],T)
    energies = calc_src_energy(dmp, T, snapshot, src_nodes=list(range(node_start,node_end)))
    y_pred.append(energies)


e0 = time.time()
print(f'{node_start}-{node_end} Elapsed Time: {e0-s0}')

y_pred = sparse.csr_matrix(np.array(y_pred))
with open(out_file, 'wb') as f:
    pickle.dump(y_pred, f)
