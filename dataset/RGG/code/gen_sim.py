import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import numpy as np
import pickle
from scipy.sparse.linalg import eigs

import sys

np.random.seed(42)


# get the snapshot at time step
# iterations is an object returned by model.iteration_bunch()
def get_snapshot(iterations, step):
    if step >= len(iterations):
        raise Exception('step must be within len(iterations)')

    g = iterations[0]['status']
    for i in range(1, step+1):
        s = iterations[i]['status']
        if len(s) > 0:
            for k in s:
                g[k] = s[k]

    return g

# save a snapshot as features
# output format: node_id, features, label
# for SIR, features are S, I, R
# label: 0 not source, 1 source
def save_features(s, src, out_file, model='SIR'):
    if model == 'SIR':
        dd = {0: '1 0 0', 1: '0 1 0', 2: '0 0 1'}
    else:
        raise Exception('unknown model')

    f = open(out_file, 'w')

    for k in sorted(s):
        if k == src:
            f.write(str(k) + ' ' + dd[s[k]] + ' ' + '1' + '\n')
        else:
            f.write(str(k) + ' ' + dd[s[k]] + ' ' + '0' + '\n')

    f.close()

# Input:
# s - snapshot
# return:
# x - data matrix of the snapshot, shape: [N, num_channels]
def getX(s, model='SIR'):
    if model == 'SIR':
        dd = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
    else:
        raise Exception('unknown model')

    x = []
    for k in sorted(s):
        x.append(dd[s[k]])

    return x

# -------------
# parameters
# --------------

para_type = 1 # 0: by (alpha,) beta, gamma; 1: by R0, gamma, (alpha)

N = 1000 # number of nodes
r = 0.08 # radius, two nodes are joined by an edge if the distance between them is at most radius


graph_id = int(sys.argv[1]) #0

gpath = '../data/graph/'
spath = '../data/SIR/'

graph_type = 'RGG'
graph_name = gpath + '%s_N%s_r%s_g%s.edgelist' % (graph_type,N,r,graph_id)

# load graph
g = nx.read_edgelist(graph_name, nodetype=int)


sim_type = 'SIR' # simulation type

T = 30 # upbound of iterations

len_seq = 2000 # num of sequences = num of sources
num_frames = 10 # num of frames = num of snapshots


if para_type == 0: # beta, gamma
    beta = 0.1 # SIR beta
    gamma = 0.1 # SIR gamma
    sim_file = '%s_beta%s_gamma%s_T%s_ls%s_nf%s_N%s_r%s_g%s.pickle' %\
    (sim_type,beta,gamma,T,len_seq,num_frames,N,r,graph_id)
elif para_type == 1: # R0, gamma
    R0 = 2.5
    gamma = 0.4
    A = nx.adjacency_matrix(g).todense().astype(float)
    lambda_max = eigs(A, k=1, which='LR')[0][0].real
    beta = round(R0 * gamma / lambda_max, 3)
    sim_file = '%s_Rzero%s_gamma%s_T%s_ls%s_nf%s_N%s_r%s_g%s.pickle' %\
    (sim_type,R0,gamma,T,len_seq,num_frames,N,r,graph_id)
else:
    raise Exception('unknown parameter type')

print('beta {} gamma {}'.format(beta, gamma))

# sequence
X = [] # features, shape [len_seq, num_frames, N, num_channels]
y = [] # labels, i.e. source node index, shape [len_seq, 1]

# simulation from different sources
for i in range(len_seq):

    src = np.random.randint(N) # a random src
    # Model Selection
    model = ep.SIRModel(g)


    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_initial_configuration('Infected', [src]) # set the source

    model.set_initial_status(config)

    # Simulation
    iterations = model.iteration_bunch(T)
    trends = model.build_trends(iterations)


    # select num_frames random frames
    ind_frames = sorted(np.random.choice(range(1,T),num_frames,replace=False))

    x_tmp = [] # shape: [num_frames, N, num_channels]

    # sample snapshots at different steps
    for j in ind_frames:

        s = get_snapshot(iterations, j)

        xs = getX(s,model=sim_type)

        x_tmp.append(xs)

        #feature_file = 'SIR_src%s_T%s_t%s_N%s_p%s_g%s.features' % (src, T, t, N, p, graph_id)

        #save_features(s, src, feature_file)

    X.append(x_tmp)
    y.append(src)

pickle.dump((np.array(X),np.array(y)), open(spath+sim_file,'wb'))

