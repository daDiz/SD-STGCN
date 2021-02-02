from data_loader.data_utils import *
from utils.math_utils import evaluation
from utils.metric_utils import *
from os.path import join as pjoin
import tensorflow as tf
import numpy as np
import time



def model_test(inputs, args, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    inputs: instance of class Dataset, data source for test.
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

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')[0]


        acc_test_list = []
        mrr_test_list = []
        hit_5_list = []
        hit_10_list = []
        hit_20_list = []
        for (x_test, y_test) in gen_xy_batch(inputs.get_data('test'), batch_size,\
        dynamic_batch=True, shuffle=False):
            x_test_ = onehot(iteration2snapshot(x_test, n_frame, start=start, end=end, random=random),n_channel)
            pred_test = test_sess.run(pred, feed_dict={'data_input:0': x_test_, 'data_label:0': onehot(y_test, num_node), 'keep_prob:0': 1.0})

            acc_test_list.append(batch_acc(pred_test, y_test))

            mrr_test_list.append(batch_mrr(pred_test, y_test))

            hit_5_list += hit_at(pred_test, y_test, 5)
            hit_10_list += hit_at(pred_test, y_test, 10)
            hit_20_list += hit_at(pred_test, y_test, 20)


        acc_test = np.mean(acc_test_list)
        mrr_test = np.mean(mrr_test_list)
        hit_5 = np.mean(hit_5_list)
        hit_10 = np.mean(hit_10_list)
        hit_20 = np.mean(hit_20_list)
        print(f'{acc_test:.3f} {mrr_test:.3f} {hit_5:.3f} {hit_10:.3f} {hit_20:.3f}')




