import time
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

class Dataset(object):
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def get_x(self, type):
        return self.__x[type]

    def get_y(self, type):
        return self.__y[type]

    def get_data(self, type):
        return (self.__x[type], self.__y[type])

    def get_len(self, type):
        return len(self.__x[type])



def data_gen(file_path, n_node, train_pct=0.8, val_pct=0.1, shuffle=True):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param n_node: int, the number of nodes in the graph.
    :return: dict, dataset that contains training, validation and test.
    '''
    # generate training, validation and test data
    try:
        data = pickle.load(open(file_path,'rb'))
        # data[0] # snapshots a list of #len_seq iterations
        # data[1] # source node index [len_seq, 1]
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    n_tot = len(data[0])

    ind_list = list(range(n_tot))

    if shuffle:
        np.random.shuffle(ind_list)

    n_train = int(n_tot * train_pct)
    n_val = int(n_tot * val_pct)

    x_train = np.array(data[0])[ind_list[:n_train]]
    y_train = np.array(data[1])[ind_list[:n_train]]

    x_val = np.array(data[0])[ind_list[n_train:(n_train+n_val)]]
    y_val = np.array(data[1])[ind_list[n_train:(n_train+n_val)]]

    x_test = np.array(data[0])[ind_list[(n_train+n_val):]]
    y_test = np.array(data[1])[ind_list[(n_train+n_val):]]


    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    y_data = {'train': y_train, 'val': y_val, 'test': y_test}

    dataset = Dataset(x_data, y_data)

    return dataset


def gen_xy_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: (x,y)
    :x: np.ndarray, [len_seq, n_frame, n_node, n_feat]
    :y: np.ndarray, [len_seq, 1]
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    x = inputs[0]
    y = inputs[1]

    len_inputs = len(x)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield (x[slide], y[slide])



# x_batch: len_seq sparse_matrix
# sparse_matrix has shape [n_frame, N]
# return: dense array, [len_seq, n_frame, N]
#def sparse2dense(x_batch):
#    return np.array([a.todense().tolist() for a in x_batch])


# sample one snapshot at ind
def sample_at(iterations, ind):
    n = len(iterations)
    if ind > n:
        raise Exception('step must be < len(iterations)')

    m = min(ind+1,n)

    g = iterations[0]['status'].copy()
    N = len(g)
    for step in range(1,m):
        s = iterations[step]['status']
        if len(s) > 0:
            for k in s:
                g[k] = s[k]

    snapshot = [g[k] for k in range(N)]

    return snapshot


# sample one snapshot from an iteration
def sample_one_snapshot(iteration, start, end=-1, random=True, ind=None):
    if end==-1:
        end = len(iteration)
    else:
        end = min(end, len(iteration))

    if random:
        ind = np.random.choice(range(start,end))
    elif ind == None:
        raise Exception('Must provide ind if random is False')
    #elif ind < start or (end > 0 and ind > end):
    #    raise Exception('ind must be > start and < end')
    elif ind > 0 and ind < 1: # percentage
        ind = int(len(iteration) * ind)

    res = sample_at(iteration, ind)

    return res



# sample one snapshot per iteration in a batch
def iteration2snapshot(iterations, start, end, random=True, ind=None):
    return np.array([sample_one_snapshot(x, start, end, random, ind) for x in iterations])


def onehot(a, n, axis=-1, dtype=int):
    pos = axis if axis >= 0 else a.ndim + axis + 1
    shape = list(a.shape)
    #shape.insert(pos, a.max() + 1)
    shape.insert(pos, n)
    out = np.zeros(shape, dtype)
    ind = list(np.indices(a.shape, sparse=True))
    ind.insert(pos, a)
    out[tuple(ind)] = True
    return out


# ------------------------
# for real world cases
# -------------------------

def single_data_gen(file_path):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :return: dict, dataset that contains training, validation and test.
    '''
    # generate training, validation and test data
    try:
        data = pickle.load(open(file_path,'rb'))
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    x_data = {'test': data[0]}
    y_data = {'test': data[1]}
    #y_data = {'test': [0]}

    dataset = Dataset(x_data, y_data)

    return dataset




if __name__=='__main__':
    sfile ='../data/ER/SIR/SIR_Rzero1-15_gamma0.1-0.4_ls2000_nf16_N1000_p0.02_g0_entire.pickle'
    n = 1000
    train_pct, val_pct = 0.8, 0.1
    batch_size = 16
    random = True
    n_channel = 3

    start = 0
    end = -1

    dataset = data_gen(sfile, n, train_pct, val_pct)

    for x_batch, y_batch in gen_xy_batch(dataset.get_data('train'), batch_size, dynamic_batch=True,\
    shuffle=True):
        x_batch_ = onehot(iteration2snapshot(x_batch, start=start, end=end, random=random),\
        n_channel)

        print(x_batch_)
        print(y_batch)






