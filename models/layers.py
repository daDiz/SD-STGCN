import tensorflow as tf

# -------------------
# gconv
# --------------------
# graph convolution using chebyshev polynomials
#
def gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    x: tensor, [batch_size, N, c_in].
    theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    Ks: int, kernel size of graph convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    return: tensor, [batch_size, N, c_out].
    '''
    # graph kernel: tensor, [N, Ks*N]
    kernel = tf.compat.v1.get_collection('graph_kernel')[0]
    n = tf.shape(input=kernel)[0]
    # x -> [batch_size, c_in, N] -> [batch_size*c_in, N]
    x_tmp = tf.reshape(tf.transpose(a=x, perm=[0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*N] -> [batch_size, c_in, Ks, N]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, N, c_in, K_s] -> [batch_size*N, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(a=x_mul, perm=[0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*N, c_out] -> [batch_size, N, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


# -----------------------
# gcn
# ----------------------
# gcn (kipf)
#
def gcn(x, Ks, C_0):
    '''
    x: tensor, [batch_size, n, C_0]
    Ks: int, kernel size of graph convolution
    C_0: int, num of channels
    return: tensor, [batch_size, n, C_0]
    '''

    # kernel -> [n, n]
    kernel = tf.compat.v1.get_collection('graph_kernel')[0]
    n = tf.shape(input=kernel)[0]

    x_gconv = x

    for i in range(Ks):
        ws = tf.compat.v1.get_variable(name='ws%d' % (i), shape=[C_0, C_0], dtype=tf.float32)

        # weight decay for the 1st layer
        if (i == 0):
            tf.compat.v1.add_to_collection(name='weight_decay%d' % (i), value=tf.nn.l2_loss(ws))

        #variable_summaries(ws, 'theta%d' % (i))
        bs = tf.compat.v1.get_variable(name='bs%d' % (i), initializer=tf.zeros([C_0]), dtype=tf.float32)

        # [batch_size, n, C_0] -> [batch_size*n, C_0]
        x_a = tf.reshape(x_gconv, [-1, C_0])
        # [batch_size*n, C_0] -> [batch_size, n, C_0]
        x_b = tf.reshape(tf.matmul(x_a, ws), [-1, n, C_0])
        # [batch_size, n, C_0] -> [batch_size, C_0, n] -> [batch_size*C_0, n]
        x_c = tf.reshape(tf.transpose(a=x_b, perm=[0,2,1]), [-1, n])
        # [batch_size*C_0, n] -> [batch_size, C_0, n]
        x_d = tf.reshape(tf.matmul(x_c, kernel), [-1, C_0, n])
        # [batch_size, C_0, n] -> [batch_size, n, C_0]
        x_e = tf.transpose(a=x_d, perm=[0,2,1])

        if i < (Ks-1):
            x_gconv = tf.nn.relu(x_e + bs)
        else:
            x_gconv = x_e + bs

    return x_gconv



def layer_norm(x, scope):
    '''
    Layer normalization function.
    x: tensor, [batch_size, time_step, N, channel].
    scope: str, variable scope.
    return: tensor, [batch_size, time_step, N, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x=x, axes=[2, 3], keepdims=True)

    with tf.compat.v1.variable_scope(scope):
        gamma = tf.compat.v1.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.compat.v1.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    x: tensor, [batch_size, time_step, N, c_in].
    Kt: int, kernel size of temporal convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    act_func: str, activation function.
    return: tensor, [batch_size, time_step-Kt+1, N, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.compat.v1.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(input=x, filters=w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(input=x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]


    if act_func == 'GLU':
        # gated linear unit
        wt = tf.compat.v1.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.compat.v1.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(input=x, filters=wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.compat.v1.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.compat.v1.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(input=x, filters=wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')

# ------------------------------
# spato_conv_layer_cheb
# ------------------------------
# spatio convolution using Chebyshev polynomials
#
def spatio_conv_layer_cheb(x, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    x: tensor, [batch_size, time_step, N, c_in].
    Ks: int, kernel size of spatial convolution.
    c_in: int, size of input channel.
    c_out: int, size of output channel.
    return: tensor, [batch_size, time_step, N, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.compat.v1.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(input=x, filters=w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(input=x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.compat.v1.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.compat.v1.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, N, c_in] -> [batch_size*time_step, N, c_out]
    x_gconv = gconv(tf.reshape(x_input, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, N, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


# ------------------------------
# spatio_conv_layer_gcn
# ------------------------------
# spatio convolution using gcn
# ----------------------------
def spatio_conv_layer_gcn(x, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.compat.v1.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(input=x, filters=w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(input=x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # x -> [batch_size*time_step, n, c_in] -> [batch_size*time_step, n, c_in]
    x_gconv = gcn(tf.reshape(x_input, [-1, n, c_in]), Ks, c_out)


    # x_g -> [batch_size, time_step, N, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])

    return tf.nn.relu(x_gc + x_input)


def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, sconv, act_func='GLU'):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    x: tensor, batch_size, time_step, N, c_in].
    Ks: int, kernel size of spatial convolution.
    Kt: int, kernel size of temporal convolution.
    channels: list, channel configs of a single st_conv block.
    scope: str, variable scope.
    keep_prob: placeholder, prob of dropout.
    sconv: type of spatio-convolution, cheb or gcn
    act_func: str, activation function.
    return: tensor, [batch_size, time_step, N, c_out].
    '''
    c_si, c_t, c_oo = channels

    with tf.compat.v1.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
        if sconv == 'cheb':
            x_t = spatio_conv_layer_cheb(x_s, Ks, c_t, c_t)
        elif sconv == 'gcn':
            x_t = spatio_conv_layer_gcn(x_s, Ks, c_t, c_t)
        else:
            raise Exception('unknown spatio-conv method')

    with tf.compat.v1.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)

    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, 1 - (keep_prob))


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    x: tensor, [batch_size, 1, N, channel].
    n: int, number of nodes / size of graph.
    channel: channel size of input x.
    scope: str, variable scope.
    return: tensor, [batch_size, 1, N, 1].
    '''
    w = tf.compat.v1.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.compat.v1.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.compat.v1.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(input=x, filters=w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    x: tensor, [batch_size, time_step, n_node, channel].
    T: int, kernel size of temporal convolution.
    scope: str, variable scope.
    act_func: str, activation function.
    return: tensor, [batch_size, n_node].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.compat.v1.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.compat.v1.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)

    # reshape [-1, n]
    x_fc = tf.reshape(x_fc, [-1,n])
    return x_fc


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    var: tf.Variable().
    v_name: str, name of the variable.
    '''
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar(f'mean_{v_name}', mean)

        with tf.compat.v1.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar(f'stddev_{v_name}', stddev)

        tf.compat.v1.summary.scalar(f'max_{v_name}', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar(f'min_{v_name}', tf.reduce_min(input_tensor=var))

        tf.compat.v1.summary.histogram(f'histogram_{v_name}', var)
