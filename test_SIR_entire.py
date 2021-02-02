# --------------
# test.py
# --------------

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join as pjoin

import tensorflow as tf

tf.compat.v1.disable_eager_execution() # disable eager execution

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.valider import model_valid
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_node', type=int, default=100)

parser.add_argument('--n_frame', type=int, default=16)

parser.add_argument('--n_channel', type=int, default=3)

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)

parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)

parser.add_argument('--sconv', type=str, default='cheb') # spatio-convolution method, cheb or gcn
                                                            # cheb --chebyshev polinomials
                                                            # gcn -- kipf's gcn

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')

parser.add_argument('--dropout', type=float, default=0)

parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--seq', type=str, default='default')


parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=-1)

parser.add_argument('--gt', type=str, default='ER')

parser.add_argument('--valid', type=int, default=0)

parser.add_argument('--random', type=int, default=1)


args = parser.parse_args()
print(f'Training configs: {args}')

n, n_frame = args.n_node, args.n_frame
Ks, Kt = args.ks, args.kt

sconv = args.sconv

# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[3, 36, 144], [144, 36, 72]]

# Load weighted adjacency matrix W
if args.graph == 'default':
    gfile = './dataset/ER/data/graph/ER_N1000_p0.02_g0.edgelist'
else:
    gfile = args.graph



# load customized graph weight matrix
W = weight_matrix(gfile)

if sconv == 'cheb':
    # Calculate graph kernel
    L = scaled_laplacian(W)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, Ks, n)
elif sconv == 'gcn':
    Lk = first_approx(W, n)
else:
    raise Exception('unknown spatio-conv method')


tf.compat.v1.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
train_pct, val_pct = 0.8, 0.1

if args.seq == 'default':
    sfile =\
    './dataset/ER/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_p0.02_g0.pickle'
else:
    sfile = args.seq


save_path='./output/models/%s/exp2/' % (args.gt)
load_path='./output/models/%s/exp2/' % (args.gt)

dataset = data_gen(sfile, n, n_frame, train_pct, val_pct)

do_train = False
do_valid = False
do_test = True

if __name__ == '__main__':
    if do_train:
        model_train(dataset, blocks, args, save_path=save_path)
    if do_valid:
        model_valid(dataset, args, load_path=load_path)
    if do_test:
        model_test(dataset, args, load_path=load_path)


