from sim_utils import *
import sys
import argparse

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--n_node', type=int, default=1000)
parser.add_argument('--n_frame', type=int, default=16)
parser.add_argument('--len_seq', type=int, default=2000)
parser.add_argument('--graph_id', type=int, default=0)
parser.add_argument('--graph_type', type=str, default='ER')
parser.add_argument('--sim_type', type=str, default='SIR')
parser.add_argument('--R0', type=float, default=2.5)
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--r', type=float, default=0.08)
parser.add_argument('--f0', type=float, default=0.02)

args = parser.parse_args()


r = args.r

graph_id = args.graph_id

N = args.n_node

sim_type = args.sim_type # simulation type


gpath = '../data/graph/'
spath = '../data/%s/' % (sim_type)

graph_type = args.graph_type
graph_name = gpath + '%s_N%s_r%s_g%s.edgelist' % (graph_type,N,r,graph_id)

# load graph
g = nx.read_edgelist(graph_name, nodetype=int)


len_seq = args.len_seq # num of sequences = num of sources
num_frames = args.n_frame # num of frames = num of snapshots

# minimum outbreak fraction
f0 = args.f0
R0 = round(args.R0, 2) #2.5
gamma = round(args.gamma, 2) #0.4 is the one used by the finding patient zero paper
alpha = round(args.alpha, 2)

A = nx.adjacency_matrix(g).todense().astype(float)
lambda_max = eigs(A, k=1, which='LR')[0][0].real
beta = round(R0 * gamma / lambda_max, 3)
if sim_type == 'SIR':
    sim_file = '%s_Rzero%s_gamma%s_ls%s_nf%s_N%s_r%s_g%s_entire.pickle' %\
    (sim_type,R0,gamma,len_seq,num_frames,N,r,graph_id)
elif sim_type == 'SEIR':
    sim_file = '%s_Rzero%s_gamma%s_alpha%s_ls%s_nf%s_N%s_r%s_g%s_entire.pickle' %\
    (sim_type,R0,gamma,alpha,len_seq,num_frames,N,r,graph_id)
else:
    raise Exception('unknown simulation type')

# sequence
X = [] # features, shape [len_seq, num_frames, N, num_channels]
y = [] # labels, i.e. source node index, shape [len_seq, 1]

# simulation from different sources
i = 0
sim_length = []
#len_seq = 1
while i < len_seq:
    if sim_type == 'SIR':
        sim = SIR(N, g, beta, gamma, min_outbreak_frac=f0)
    elif sim_type == 'SEIR':
        sim = SEIR(N, g, beta, gamma, alpha, min_outbreak_frac=f0)
    else:
        raise Exception('uknown simulation type')

    sim.init()
    sim.run()


    if sim.is_outbreak and len(sim.iterations) > num_frames:
        # select num_frames random frames
        #ind_frames = sorted(np.random.choice(range(1,len(sim.iterations)),num_frames,replace=False))

        #x_tmp = [] # shape: [num_frames, N, num_channels]

        # sample snapshots at different steps
        #for j in ind_frames:

        #    s = sim.get_snapshot(j)

        #    x_tmp.append(s)
        sim_length.append(len(sim.iterations))

        i += 1
        #X.append(sparse.csr_matrix(x_tmp))
        X.append(sim.iterations)
        y.append(sim.src)

pickle.dump((X,y), open(spath+sim_file,'wb'))

print('-----------------------------------------------------')
print('mean epidemic length: %.1f' % (np.mean(sim_length)))
print('stdev epidemic length: %.2f' % (np.std(sim_length)))
print('-----------------------------------------------------')

