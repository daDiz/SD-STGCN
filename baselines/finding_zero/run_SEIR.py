import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import *
from pygcn.models import FZGCN
from pygcn.data_loader import *
from pygcn.metric_utils import *

import argparse

# ------------------
# check device
# ------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# ---------------------
# parameters
# ----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--n_node', type=int, default=100)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)


parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--dropout', type=float, default=0)

parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--seq_train', type=str, default='default')
parser.add_argument('--seq_test', type=str, default='default')

parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=-1)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--save', type=str, default=None)

args = parser.parse_args()
#print(f'Training configs: {args}')


train_pct, val_pct = 0.8, 0.1
batch_size = args.batch_size

random = True # whether to sample a snapshot randomly

start = args.start # lower bound to sample
end = args.end # upper bound to sample

N = args.n_node

n_channel = args.n_channel # num of channels (SIR - 3, SEIR - 4)
n_hidden = args.n_hidden
n_gcn = args.n_layer # num of gcn layers

dropout = args.dropout

# learning rate related
lr = args.lr
decay_factor = 0.5
patience = 10

mode = args.mode
save_path = args.save

n_epoch = args.epoch

# -------------------
# load adj
# --------------------
g_path = args.graph

adj = load_adj(g_path).to(device)

# -------------------
# load snapshots
# --------------------
train_path = args.seq_train
test_path = args.seq_test

if mode == 'train':
    dataset = data_gen(train_path, N, train_pct, val_pct)
elif mode == 'test':
    dataset = data_gen(test_path, N, train_pct, val_pct)
else:
    raise Exception('unknown mode: train or test')


# Model and optimizer
model = FZGCN(nin=n_channel,
            nhid=n_hidden,
            nnode=N,
            nlayer=n_gcn,
            dropout=dropout)

#print(model)

model.to(device) # model to device

optimizer = optim.Adam(model.parameters(),
                       lr=lr)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', \
                                        factor=decay_factor,
                                        patience=patience)


def train(epoch):
    model.train()
    #for epoch in range(n_epoch):

    for it, (features, labels) in enumerate(gen_xy_batch(dataset.get_data('train'), batch_size,\
    dynamic_batch=True, shuffle=True)):
        t = time.time()

        features_ = torch.tensor(onehot(iteration2snapshot(features, start=start, end=end,\
        random=random), n_channel),dtype=torch.float).to(device)

        labels_ = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        output = model(features_, adj)
        loss_train = F.nll_loss(output, labels_)
        acc_train = accuracy(output, labels_)
        loss_train.backward()
        optimizer.step()

        if it % 100 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                  'Iter: {:04d}'.format(it+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t))

def validate(epoch):
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    t = time.time()
    loss_list = []
    acc_list = []
    for it, (features, labels) in enumerate(gen_xy_batch(dataset.get_data('val'), batch_size,\
    dynamic_batch=True, shuffle=True)):

        features_ = torch.tensor(onehot(iteration2snapshot(features, start=start, end=end,\
        random=random), n_channel),dtype=torch.float).to(device)

        labels_ = torch.tensor(labels).to(device)

        output = model(features_, adj)
        loss_list.append(F.nll_loss(output, labels_).item())
        acc_list.append(accuracy(output, labels_).item())

    loss_val = np.mean(loss_list)
    acc_val = np.mean(acc_list)
    print('Valid: {:04d}'.format(epoch+1),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))
    print('---------------------------')

    return loss_val

def test():
    # Evaluate test set performance separately,
    # deactivates dropout during validation run.
    model.load_state_dict(torch.load(save_path))
    model.eval()
    acc_list = []
    mrr_list = []
    hit5_list = []
    hit10_list = []
    hit20_list = []

    for it, (features, labels) in enumerate(gen_xy_batch(dataset.get_data('test'), batch_size,\
    dynamic_batch=True, shuffle=False)):

        features_ = torch.tensor(onehot(iteration2snapshot(features, start=start, end=end,\
        random=random), n_channel),dtype=torch.float).to(device)

        labels_ = torch.tensor(labels).to(device)

        output = model(features_, adj)

        y_pred = output.cpu().detach().numpy()
        y_true = labels_.cpu().detach().numpy()

        acc_list.append(batch_acc(y_pred, y_true))
        mrr_list.append(batch_mrr(y_pred, y_true))
        hit5_list += hit_at(y_pred, y_true, top=5)
        hit10_list += hit_at(y_pred, y_true, top=10)
        hit20_list += hit_at(y_pred, y_true, top=20)


    acc_test = np.mean(acc_list)
    mrr_test = np.mean(mrr_list)
    hit5_test = np.mean(hit5_list)
    hit10_test = np.mean(hit10_list)
    hit20_test = np.mean(hit20_list)
    print(f'{acc_test:.4f} {mrr_test:.4f} {hit5_test:.4f} {hit10_test:.4f} {hit20_test:.4f}')



if mode == 'train':
    t_total = time.time()
    for epoch in range(n_epoch):
        train(epoch)
        val_loss = validate(epoch)
        scheduler.step(val_loss)

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    torch.save(model.state_dict(), save_path)

elif mode == 'test':
    test()
else:
    raise Exception('unknown mode: train or test')

