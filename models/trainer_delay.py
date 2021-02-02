from data_loader.data_utils_delay import *
from utils.metric_utils import *
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time


def model_train(inputs, blocks, args, pct_list, save_path='./output/models/', sum_path='./output/tensorboard'):
    '''
    Train the base model.
    inputs: instance of class Dataset, data source for training.
    blocks: list, channel configs of st_conv blocks.
    args: instance of class argparse, args for training.
    '''
    n, n_frame, n_channel = args.n_node, args.n_frame, args.n_channel
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, opt = args.batch_size, args.epoch, args.opt

    sconv = args.sconv # spatio-conv type

    dropout = args.dropout

    start, end = args.start, args.end

    valid = args.valid

    random = args.random

    # Placeholder for model training
    x = tf.compat.v1.placeholder(tf.float32, [None, n_frame, n, n_channel], name='data_input')

    # placeholder for one-hot labels
    y = tf.compat.v1.placeholder(tf.float32, [None, n], name='data_label')

    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    # Define model loss
    train_loss, pred = build_model(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv)
    tf.compat.v1.summary.scalar('train_loss', train_loss)


    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')

    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1

    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.compat.v1.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.compat.v1.summary.scalar('learning_rate', lr)
    step_op = tf.compat.v1.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.compat.v1.summary.merge_all()



    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(epoch):
            start_time = time.time()
            for j, (x_batch, y_batch) in enumerate(
                gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):

                for start_pct in pct_list:
                    x_batch_ = onehot(iterationSnapshotSlide(x_batch, n_frame, start_pct=start_pct), n_channel)

                    summary, _ = sess.run([merged, train_op],\
                                feed_dict={x: x_batch_,\
                                y: onehot(y_batch, n), keep_prob: 1.0})


                if j % 10 == 0:
                    loss_value, pred_batch = \
                        sess.run([train_loss,pred],
                                 feed_dict={x: x_batch_, y: onehot(y_batch,n), keep_prob: 1-dropout})

                    acc = batch_acc(pred_batch, y_batch)

                    if valid:
                        # evaluate acc on validation
                        acc_val_list = []
                        for (x_val, y_val) in gen_xy_batch(inputs.get_data('val'), batch_size, dynamic_batch=True, shuffle=False):

                            x_val_ = onehot(iteration2snapshot(x_val,n_frame,start=start,end=end,start_pct=start_pct, random=random), n_channel)


                            pred_val = sess.run(pred, feed_dict={x: x_val_, y:\
                                                        onehot(y_val, n), keep_prob: 1.0})
                            acc_val_list.append(batch_acc(pred_val, y_val))


                        acc_val = np.mean(acc_val_list)
                        end_time = time.time()
                        print(f'Epoch {i:2d}, Step {j:3d}: Elapsed Time {(end_time-start_time):.3f} Loss {loss_value:.3f} Train Acc {acc:.3f} Val Acc {acc_val:.3f}')
                    else:
                        end_time = time.time()
                        print(f'Epoch {i:2d}, Step {j:3d}: Elapsed Time {(end_time-start_time):.3f} Loss {loss_value:.3f} Train Acc {acc:.3f}')


            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'STGCN', save_path)

    print('Training model finished!')
