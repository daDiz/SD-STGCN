# -------------------------------------
# simulations with mix R0, gamma randomly picked from given ranges
# simulations end when I = 0
# -----------------------------------
from sim_utils import *
import sys
import argparse

np.random.seed(42)


R0_lo = 1.0
R0_hi = 15.0
gamma_lo = 0.1
gamma_hi = 0.4

R0_k = '1-15'
gamma_k = '0.1-0.4'

parser = argparse.ArgumentParser()
parser.add_argument('--n_node', type=int, default=1000)
parser.add_argument('--n_frame', type=int, default=16)
parser.add_argument('--len_seq', type=int, default=2000)
parser.add_argument('--graph_id', type=int, default=0)
parser.add_argument('--graph_type', type=str, default='ER')
parser.add_argument('--sim_type', type=str, default='SIR')
parser.add_argument('--r', type=float, default=0.08)
parser.add_argument('--f0', type=float, default=0.02)

args = parser.parse_args()


r = args.r # random graph r for RGG

graph_id = args.graph_id

N = args.n_node

sim_type = args.sim_type # simulation type


gpath = '../data/graph/'
spath = '../data/%s/' % (sim_type)

graph_type = args.graph_type
graph_name = gpath + '%s_N%s_r%s_g%s.edgelist' % (graph_type,N,r,graph_id)

# load graph
g = nx.read_edgelist(graph_name, nodetype=int)

A = nx.adjacency_matrix(g).todense().astype(float)
lambda_max = eigs(A, k=1, which='LR')[0][0].real

len_seq = args.len_seq # num of sequences = num of sources
num_frames = args.n_frame # num of frames = num of snapshots

# minimum outbreak fraction
f0 = args.f0
#alpha = round(args.alpha, 2)


sim_file = '%s_Rzero%s_gamma%s_ls%s_nf%s_N%s_r%s_g%s_entire.pickle' %\
            (sim_type,R0_k,gamma_k,len_seq,num_frames,N,r,graph_id)

# sequence
X = [] # features, shape [len_seq, num_frames, N, num_channels]
y = [] # labels, i.e. source node index, shape [len_seq, 1]

# simulation from different sources
i = 0
sim_length = []
while i < len_seq:

    R0 = np.random.uniform(R0_lo, R0_hi)
    gamma = np.random.uniform(gamma_lo, gamma_hi)
    beta = R0 * gamma / lambda_max

    sim = SIR(N, g, beta, gamma, min_outbreak_frac=f0)
    sim.init()
    sim.run()

    if sim.is_outbreak and len(sim.iterations) > num_frames:
        sim_length.append(len(sim.iterations))

        i += 1
        X.append(sim.iterations)
        y.append(sim.src)

print('-----------------------------------------------------')
print('mean epidemic length: %.1f' % (np.mean(sim_length)))
print('stdev epidemic length: %.2f' % (np.std(sim_length)))
print('-----------------------------------------------------')

pickle.dump((X,y), open(spath+sim_file,'wb'))


