from data_loader.data_utils import *
from utils.math_utils import evaluation
from utils.metric_utils import *
from os.path import join as pjoin
import tensorflow as tf
import numpy as np
import time



def model_valid(inputs, args, load_path='./output/models/'):
    '''
    Load and valid saved model from the checkpoint.
    inputs: instance of class Dataset, data source for validation.
    args: instance of class argparse, args for training.
    load_path: str, the path of loaded model.
    '''
    n_frame = args.n_frame
    num_node = args.n_node
    n_channel = args.n_channel

    batch_size = args.batch_size

    start, end = args.start, args.end

    random = args.random

    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    valid_graph = tf.Graph()

    with valid_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=valid_graph) as valid_sess:
        saver.restore(valid_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = valid_graph.get_collection('y_pred')[0]


        acc_valid_list = []
        mrr_valid_list = []
        hit_5_list = []
        hit_10_list = []
        hit_20_list = []
        for (x_valid, y_valid) in gen_xy_batch(inputs.get_data('val'), batch_size,\
        dynamic_batch=True, shuffle=False):
            x_valid_ = onehot(iteration2snapshot(x_valid, n_frame, start=start, end=end, random=random),n_channel)
            pred_valid = valid_sess.run(pred, feed_dict={'data_input:0': x_valid_, 'data_label:0': onehot(y_valid, num_node), 'keep_prob:0': 1.0})

            acc_valid_list.append(batch_acc(pred_valid, y_valid))

            mrr_valid_list.append(batch_mrr(pred_valid, y_valid))

            hit_5_list += hit_at(pred_valid, y_valid, 5)
            hit_10_list += hit_at(pred_valid, y_valid, 10)
            hit_20_list += hit_at(pred_valid, y_valid, 20)



        acc_valid = np.mean(acc_valid_list)
        mrr_valid = np.mean(mrr_valid_list)
        hit_5 = np.mean(hit_5_list)
        hit_10 = np.mean(hit_10_list)
        hit_20 = np.mean(hit_20_list)
        print(f'{acc_valid:.3f} {mrr_valid:.3f} {hit_5:.3f} {hit_10:.3f} {hit_20:.3f}')



