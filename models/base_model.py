from models.layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv):
    '''
    Build the base model.
    x: placeholder features, [-1, n_frame, n, n_channel]
    y: placeholder label, [-1, n]
    n_frame: int, size of records for training.
    Ks: int, kernel size of spatial convolution.
    Kt: int, kernel size of temporal convolution.
    blocks: list, channel configs of st_conv blocks.
    keep_prob: placeholder.
    sconv: type of spatio-convolution layer, cheb or gcn
    '''

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_frame

    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, sconv, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        # logits shape: [-1, n_node]
        logits = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')



    train_loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                            labels=y, logits=logits, axis=-1))


    y_pred = tf.nn.softmax(logits)
    tf.compat.v1.add_to_collection(name='y_pred', value=y_pred)

    return train_loss, y_pred


def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    sess: tf.Session().
    global_steps: tensor, record the global step of training in epochs.
    model_name: str, the name of saved model.
    save_path: str, the path of saved model.
    '''
    saver = tf.compat.v1.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
