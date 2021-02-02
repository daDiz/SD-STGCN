import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GCN
import torch

# finding zero gnn
class FZGCN(nn.Module):
    def __init__(self, nin, nhid, nnode, nlayer, dropout=0):
        '''
        nin: 3 for SIR, 4 for SEIR
        nhid: hidden dim
        '''
        super(FZGCN, self).__init__()

        self.nin = nin
        self.nhid = nhid
        self.nnode = nnode
        self.nlayer = nlayer

        self.fc1 = nn.Linear(nin, nhid)

        self.gc_list = nn.ModuleList([GCN(nhid, nhid, bias=True) for _ in range(nlayer)])
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(nnode) for _ in range(nlayer)])

        #self.gc1 = GCN(nhid, nhid, bias=True)
        #self.gc2 = GCN(nhid, nhid, bias=True)
        #self.gc3 = GCN(nhid, nhid, bias=True)
        #self.gc4 = GCN(nhid, nhid, bias=True)

        #self.bn1 = nn.BatchNorm1d(nnode)
        #self.bn2 = nn.BatchNorm1d(nnode)
        #self.bn3 = nn.BatchNorm1d(nnode)
        #self.bn4 = nn.BatchNorm1d(nnode)

        self.dropout = dropout
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)

    def forward(self, x, adj):
        x = torch.reshape(x, (-1, self.nin))
        x = self.fc1(x)
        x = torch.reshape(x, (-1, self.nnode, self.nhid))

        if self.dropout > 0:
            for i in range(self.nlayer):
                x = x + F.dropout(F.leaky_relu(self.bn_list[i](self.gc_list[i](x, adj))), self.dropout, training=self.training)
            #x = x + F.dropout(F.leaky_relu(self.bn1(self.gc1(x, adj))), self.dropout, training=self.training)
            #x = x + F.dropout(F.leaky_relu(self.bn2(self.gc2(x, adj))), self.dropout, training=self.training)
            #x = x + F.dropout(F.leaky_relu(self.bn3(self.gc3(x, adj))), self.dropout, training=self.training)
            #x = x + F.dropout(F.leaky_relu(self.bn3(self.gc3(x, adj))), self.dropout, training=self.training)
            #x = x + F.dropout(F.leaky_relu(self.bn4(self.gc4(x, adj))), self.dropout, training=self.training)
        else:
            for i in range(self.nlayer):
                x = x + F.leaky_relu(self.bn_list[i](self.gc_list[i](x, adj)))
            #x = x + F.leaky_relu(self.bn1(self.gc1(x, adj)))
            #x = x + F.leaky_relu(self.bn2(self.gc2(x, adj)))
            #x = x + F.leaky_relu(self.bn3(self.gc3(x, adj)))
            #x = x + F.leaky_relu(self.bn4(self.gc4(x, adj)))


        x = torch.reshape(x, (-1, self.nhid))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1, self.nnode))

        return F.log_softmax(x, dim=1)



