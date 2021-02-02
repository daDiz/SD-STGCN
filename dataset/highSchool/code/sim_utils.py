import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import numpy as np
import pickle
from scipy.sparse.linalg import eigs
from scipy import sparse

import sys

np.random.seed(42)

class Epidemic():
    def __init__(self, N, g, min_outbreak_frac=0.02):
        self.N = N
        self.g = g
        #self.beta = beta
        #self.gamma = gamma
        self.model = None
        self.src = None
        self.infected = set([])
        self.iterations = []
        self.I = 1
        self.cur_state = np.zeros(N)
        self.min_outbreak_frac = min_outbreak_frac
        self.min_outbreak_size = int(N * min_outbreak_frac)

    def init(self):
        pass

    def iteration(self):
        self.iterations.append(self.model.iteration())
        cur = self.iterations[-1]['status']
        if len(cur) > 0:
            for k in cur:
                self.cur_state[k] = cur[k]
                if cur[k] == self.I:
                    self.infected.add(k)

    def iteration_bunch(self, T):
        self.iterations = self.model.iteration_bunch(T)
        for cur in self.iterations:
            for k in cur:
                if cur[k] == self.I:
                    self.infected.add(k)


    # return true, if the epidemic is over
    # the epidemic is over, if no one is infected
    def is_over(self):
        return all([x != self.I for x in self.cur_state])


    # run a simulation until the epidemic is over
    def run(self):
        self.iteration()
        while not self.is_over():
            self.iteration()

    # return true, if the simulation is an outbreak
    def is_outbreak(self):
        return len(self.infected) > self.min_outbreak_size


    # get the snapshot at time step
    # step 0 is the initial state with the source
    # step 1 is the first iteration
    def get_snapshot(self, step):
        if step > len(self.iterations):
            raise Exception('step must be within len(iterations)')

        g = self.iterations[0]['status']
        for i in range(1, step+1):
            s = self.iterations[i]['status']
            if len(s) > 0:
                for k in s:
                    g[k] = s[k]

        return [g[k] for k in range(self.N)]



class SIR(Epidemic):
    def __init__(self, N, g, beta, gamma, min_outbreak_frac=0.02):
        super().__init__(N, g, min_outbreak_frac)
        self.beta = beta
        self.gamma = gamma

    def init(self):
        self.src = np.random.randint(self.N) # a random src
        # Model Selection
        self.model = ep.SIRModel(self.g)

        # Model Configuration
        config = mc.Configuration()
        config.add_model_parameter('beta', self.beta)
        config.add_model_parameter('gamma', self.gamma)
        config.add_model_initial_configuration('Infected', [self.src]) # set the source

        self.model.set_initial_status(config)


class SEIR(Epidemic):
    def __init__(self, N, g, beta, gamma, alpha, min_outbreak_frac=0.02):
        super().__init__(N, g, min_outbreak_frac)
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

    def init(self):
        self.src = np.random.randint(self.N) # a random src
        # Model Selection
        self.model = ep.SEIRModel(self.g)

        # Model Configuration
        config = mc.Configuration()
        config.add_model_parameter('beta', self.beta)
        config.add_model_parameter('gamma', self.gamma)
        config.add_model_parameter('alpha', self.alpha)
        config.add_model_initial_configuration('Infected', [self.src]) # set the source

        self.model.set_initial_status(config)



if __name__=='__main__':
    # -------------
    # parameters
    # --------------

    para_type = 1 # 0: by (alpha,) beta, gamma; 1: by R0, gamma, (alpha)

    #N = 1000 # number of nodes
    p = 0.02 # random graph probability to connect

    graph_id = int(sys.argv[1]) #0

    N = int(sys.argv[2])

    sim_type = 'SIR' # simulation type


    gpath = '../data/graph/'
    spath = '../data/%s/' % (sim_type)

    graph_type = 'ER'
    graph_name = gpath + '%s_N%s_p%s_g%s.edgelist' % (graph_type,N,p,graph_id)

    # load graph
    g = nx.read_edgelist(graph_name, nodetype=int)


    len_seq = 2000 # num of sequences = num of sources
    num_frames = 16 # num of frames = num of snapshots

    # minimum outbreak fraction
    f0 = 0.02
    #s0 = int(f0 * N) # minimum outbreak size
                     # if final size > s0, the simulation is an epidemic


    if para_type == 0: # beta, gamma
        beta = 0.1 # SIR beta
        gamma = 0.1 # SIR gamma
        sim_file = '%s_beta%s_gamma%s_ls%s_nf%s_N%s_p%s_g%s_sparse.pickle' %\
        (sim_type,beta,gamma,len_seq,num_frames,N,p,graph_id)
    elif para_type == 1: # R0, gamma
        R0 = round(float(sys.argv[3]), 2) #2.5
        gamma = round(float(sys.argv[4]), 2) #0.6 # 0.4 is the one used by the finding patient zero paper
                    # we also do simulations for other gamma values, 0.2, 0.6, 0.8
        A = nx.adjacency_matrix(g).todense().astype(float)
        lambda_max = eigs(A, k=1, which='LR')[0][0].real
        beta = round(R0 * gamma / lambda_max, 3)
        sim_file = '%s_Rzero%s_gamma%s_ls%s_nf%s_N%s_p%s_g%s_sparse.pickle' %\
        (sim_type,R0,gamma,len_seq,num_frames,N,p,graph_id)
    else:
        raise Exception('unknown parameter type')

    #print('beta {} gamma {}'.format(beta, gamma))

    # sequence
    X = [] # features, shape [len_seq, num_frames, N, num_channels]
    y = [] # labels, i.e. source node index, shape [len_seq, 1]

    # simulation from different sources
    i = 0
    sim_length = []
    len_seq = 1
    while i < len_seq:

        sim = SIR(N, g, beta, gamma, min_outbreak_frac=f0)
        sim.init()
        sim.run()


        if sim.is_outbreak and len(sim.iterations) > num_frames:
            #print(f'simulation length {len(iterations)}')
            # select num_frames random frames
            ind_frames = sorted(np.random.choice(range(1,len(sim.iterations)),num_frames,replace=False))

            x_tmp = [] # shape: [num_frames, N, num_channels]

            #ind_frames = [1]

            #print(get_snapshot(iterations, 0))

            # sample snapshots at different steps
            for j in ind_frames:

                s = sim.get_snapshot(j)

                #print(s)

                #xs = getX(s,model=sim_type)
                #xs = getX(s)

                #print(xs)

                x_tmp.append(s)
                sim_length.append(len(sim.iterations))


            #print(x_tmp)
            i += 1
            X.append(sparse.csr_matrix(x_tmp))
            y.append(sim.src)

    #pickle.dump((X,y), open(spath+sim_file,'wb'))

    print('-----------------------------------------------------')
    print('mean epidemic length: %.1f' % (np.mean(sim_length)))
    print('stdev epidemic length: %.2f' % (np.std(sim_length)))
    print('-----------------------------------------------------')

