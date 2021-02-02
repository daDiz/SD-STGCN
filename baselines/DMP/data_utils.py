import time
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf


np.random.seed(42)

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



def data_gen(file_path, train_pct=0.8, val_pct=0.1, shuffle=True):
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


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

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

        yield inputs[slide]


def gen_xy_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: (x,y) or (x',y,y_) for corrupt data
    :x: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :y: np.ndarray, [len_seq, 1]
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    #x, y = inputs
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
def sparse2dense(x_batch):
    return np.array([a.todense().tolist() for a in x_batch])

# ---------------------------
# new stuff
# -------------------------

# get the snapshot at time step
# step 0 is the initial state with the source
# step 1 is the first iteration
def get_one_snapshot(iterations, step):
    if step > len(iterations):
        raise Exception('step must be within len(iterations)')

    g = iterations[0]['status'].copy()
    N = len(g)
    for i in range(1, step+1):
        s = iterations[i]['status']
        if len(s) > 0:
            for k in s:
                g[k] = s[k]

    return [g[k] for k in range(N)]

# sample a list of snapshots from an iteration
def sample_snapshot(iteration, ind):
    return [get_one_snapshot(iteration, step) for step in ind]

def sample_snapshots(iterations, ind):
    n = len(iterations)
    if np.max(ind) > n:
        raise Exception('step must be with len(iterations)')

    m = max(np.max(ind)+1,n)
    snapshots = []
    g = iterations[0]['status'].copy()
    N = len(g)
    for step in range(1,m):
        s = iterations[step]['status']
        if len(s) > 0:
            for k in s:
                g[k] = s[k]

        if step in ind:
            snapshots.append([g[k] for k in range(N)])

    if len(snapshots) < len(ind):
        tmp = snapshots[-1]
        for _ in range(len(ind) - len(snapshots)):
            snapshots.append(tmp)

    return snapshots

def sample_snapshots2(iterations, ind):
    n = len(iterations)
    if np.max(ind) > n:
        raise Exception('step must be with len(iterations)')

    m = max(np.max(ind)+1,n)
    snapshots = []
    g = iterations[0]['status'].copy()
    N = len(g)
    cur = np.zeros(N, dtype=int)
    cur[list(g.keys())] = list(g.values())
    for step in range(1,m):
        s = iterations[step]['status']
        if len(s) > 0:
            cur[list(s.keys())] = list(s.values())

        if step in ind:
            snapshots.append(list(cur))

    return snapshots


# sample some snapshots from an iteration
def sample_from_iteration(iteration, num_frames, start, end=-1, random=1):
    #if end > len(iteration):
        #raise Exception('End is out of bound')
    if end==-1:
        end = len(iteration)
    else:
        end = min(end, len(iteration))

    #if (end - start) < num_frames:
    #    raise Exception('Not enough frames')

    #t0 = time.time()
    if random == 1:
        ind = sorted(np.random.choice(range(start,end),num_frames,replace=False))
    else:
        if start + num_frames > end: # pad ind with end-1
            ind = list(range(start, end)) + [end-1] * (start + num_frames - end)
        else:
            ind = list(range(start, start+num_frames))

    #ind = list(range(end-num_frames, end))
    #ind = list(range(1,num_frames+1))
    res = sample_snapshots(iteration, ind)
    return res

# sample some snapshots from a batch of iterations
def iteration2snapshot(iterations, num_frames, start, end, random=True):
    return np.array([sample_from_iteration(x, num_frames, start, end, random) for x in iterations])


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

if __name__=='__main__':
    x = np.array([1,2,3])
    print(x.shape)
    print(one_hot(x,4))
