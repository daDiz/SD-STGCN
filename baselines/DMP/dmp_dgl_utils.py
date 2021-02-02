import networkx as nx
import torch
import dgl
import dgl.function as fn
import numpy as np
import time
from data_utils import *

class DMP():
    def __init__(self, g, lamb=0.1, mu=0.4):
        self.g = dgl.DGLGraph(g)
        self.N = self.g.number_of_nodes()
        self.LAMBDA = lamb
        self.MU = mu

    def reset(self, src):
        p0 = torch.ones(self.N)
        p0[src] = 0
        pi0 = torch.zeros(self.N)
        pi0[src] = 1

        self.g.ndata['p0'] = p0
        self.g.ndata['ps'] = p0
        self.g.ndata['pr'] = torch.zeros(self.N)
        self.g.ndata['pi'] = pi0

        self.g.edata['theta'] = torch.ones(self.g.number_of_edges())
        #self.g.edata['ps_ij'] = torch.ones(g.number_of_edges())


    def init_phi(self, edges):
        return {'phi': edges.src['pi']}

    def init_ps_ij(self, edges):
        return {'ps_ij': edges.src['ps']}

    def update_ps_ij(self, edges):
        return {'ps_ij': edges.src['ps'] / edges.data['theta']}

    def update_theta(self, edges):
        return {'theta': edges.data['theta'] - self.LAMBDA * edges.data['phi']}

    def update_phi_1(self, edges):
        return {'phi': (1-self.LAMBDA)*(1-self.MU)*edges.data['phi']+edges.data['ps_ij']}

    def update_phi_2(self, edges):
        return {'phi': edges.data['phi'] - edges.data['ps_ij']}

    def reduce_func(self, nodes):
        return {'m_prod': torch.prod(nodes.mailbox['m'], dim=1)}

    def forward(self):
        self.g.apply_edges(self.update_theta)
        #print(g.edata)
        #print(g.ndata)

        #g.update_all(message_func=fn.copy_e('theta', out='m'),
        #             reduce_func=fn.prod(msg='m',out='m_prod'))

        self.g.update_all(message_func=fn.copy_e('theta', out='m'),
                            reduce_func=self.reduce_func)

        #print(g.ndata)

        self.g.ndata['ps'] = self.g.ndata['p0'] * self.g.ndata['m_prod']
        self.g.ndata['pr'] = self.g.ndata['pr'] + self.MU * self.g.ndata['pi']
        self.g.ndata['pi'] = 1 - self.g.ndata['pr'] - self.g.ndata['ps']

        #print(g.ndata)

        self.g.apply_edges(self.update_phi_1)

        #print(g.ndata)

        self.g.apply_edges(self.update_ps_ij)

        #print(g.ndata)

        self.g.apply_edges(self.update_phi_2)
        #print(g.ndata)

    # propagate for T iterations
    def propagate(self, T):
        self.g.apply_edges(self.init_phi)
        self.g.apply_edges(self.init_ps_ij)

        for k in range(T):
            self.forward()

    # snapshot is a list of node states (0 - S, 1 - I, 2 - R)
    # not really energy, it is likelihood
    def calc_energy(self, snapshot):
        tot_Ps = 1.0
        tot_Pi = 1.0
        tot_Pr = 1.0
        for i in range(len(snapshot)):
            s = snapshot[i]
            if s == 0:
                tot_Ps *= self.g.ndata['ps'][i]
            elif s == 1:
                tot_Pi *= self.g.ndata['pi'][i]
            elif s == 2:
                tot_Pr *= self.g.ndata['pr'][i]
            else:
                raise Exception("unknown state")

        Po = tot_Ps * tot_Pi * tot_Pr
        return Po
        #return -1.0 * np.log(Po)

    def propagate_calc_energy(self, dict_snapshots):
        self.g.apply_edges(self.init_phi)
        self.g.apply_edges(self.init_ps_ij)

        T = max(dict_snapshots.keys())
        energies = []
        for i in range(1,T+1):
            self.forward()
            if i in dict_snapshots.keys():
                energies.append(self.calc_energy(dict_snapshots[i]).item())

        return energies

# calc energy for the snapshot at time T
def calc_src_energy(dmp, T, snapshot, src_nodes=None):
    if src_nodes == None:
        src_nodes = dmp.g.nodes()

    energies = []
    #t1_list = []
    #t2_list = []
    #s0 = time.time()
    for src in src_nodes:
        dmp.reset(src)
        #s = time.time()
        dmp.propagate(T)
        #e = time.time()
        #t1_list.append(e-s)
        #s = time.time()
        energies.append(dmp.calc_energy(snapshot).item())
        #e = time.time()
        #t2_list.append(e-s)

    #e0 = time.time()
    #print(np.mean(t1_list))
    #print(np.mean(t2_list))
    #print(e0 - s0)

    return energies

# calc energy for a list of snapshots
def calc_energy_at(dmp, dict_snapshots, src_nodes=None):
    if src_nodes == None:
        src_nodes = dmp.g.nodes()

    energies = []
    for src in src_nodes:
        dmp.reset(src)
        tmp = dmp.propagate_calc_energy(dict_snapshots)
        energies.append(tmp)

    return energies




