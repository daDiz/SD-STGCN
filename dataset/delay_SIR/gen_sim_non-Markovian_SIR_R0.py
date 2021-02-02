import EoN
import networkx as nx
import random
import pickle
import numpy as np
from utils import *
import sys
from scipy.sparse.linalg import eigs


np.random.seed(42)

len_seq = 2000 # number of simulations

# ---------------------------------------------------
# graph parameters
# ---------------------------------------------------
N=1000
ps = 'p0.02' #'m1' #'m10' #'p0.02'
graph_id = int(sys.argv[1])
graph_type = 'ER' #'BA' # 'ER'

gpath = '../%s/data/graph/' % (graph_type)
graph_name = gpath + '%s_N%s_%s_g%s.edgelist' % (graph_type,N,ps,graph_id)

# load graph
G = nx.read_edgelist(graph_name, nodetype=int)
A = nx.adjacency_matrix(G).todense().astype(float)
lambda_max = eigs(A, k=1, which='LR')[0][0].real


# ---------------------------------------------------
# simulation parameters
# ---------------------------------------------------

# enforce a min simulation length
min_length = 30


sim_type = 'SIR'
D = 14
R0 = 2.5 # R0 = tau * lambda_max * D
tau = R0 / lambda_max / D

min_outbreak_frac = 0.02 # tot_num_infected / N > min_outbreak_frac



spath = './sim/'
sim_file = '%s_Rzero%s_D%s_N%s_%s_g%s.pickle' %\
            (sim_type,R0,D,N,ps,graph_id)


#set up the code to handle constant transmission rate
#with fixed recovery time.
def trans_time_fxn(source, target, rate):
    return random.expovariate(rate)

def rec_time_fxn(node,D):
    return D

# -------------------------------------------------------
X = [] # iterations
y = [] # source
sim_length = []
i = 0
#for i in range(len_seq):
while i < len_seq:
    src = np.random.randint(0, N) # source node (initial infected)

    initial_inf = [src]

    sim = EoN.fast_nonMarkov_SIR(G,
                            trans_time_fxn=trans_time_fxn,
                            rec_time_fxn=rec_time_fxn,
                            trans_time_args=(tau,),
                            rec_time_args=(D,),
                            initial_infecteds = initial_inf,
                            return_full_data=True)

    R = sim.R()
    if R[-1] / N > min_outbreak_frac:
        iteration = sim2Iter_unitTime(sim, G, status_dict={'S':0, 'I':1, 'R':2})

        if len(iteration) > min_length:
            sim_length.append(len(iteration))

            X.append(iteration)
            y.append(src)

            i += 1

print('-----------------------------------------------------')
print('mean epidemic length: %.1f' % (np.mean(sim_length)))
print('stdev epidemic length: %.2f' % (np.std(sim_length)))
print('min epidemic length: %d' % (min(sim_length)))
print('max epidemic length: %d' % (max(sim_length)))
print('-----------------------------------------------------')

pickle.dump((X,y), open(spath+sim_file,'wb'))


