import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNEL = 1  # 通道数
CONV1_SIZE = 5  # 卷积核长
CONV1_KERNEL = 32  # 卷积核个数
CONV2_SIZE = 5
CONV2_KERNEL = 64
FC_SIZE = 512   # 第一层全连接神经网络节点个数
OUTPUT_NODE = 10    # 输出节点个数


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def forward(x, train, regularizer):
    # 卷积第一层
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_KERNEL], regularizer)
    conv1_b = get_bias([CONV1_KERNEL])
    conv1 = conv2d(x, conv1_w)
    # 激活
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 池化
    pool1 = max_pool_2x2(relu1)

    # 卷积第二层
    conv2_w = get_weight([CONV2_SIZE, CONV1_SIZE, CONV1_KERNEL, CONV2_KERNEL], regularizer)
    conv2_b = get_bias([CONV2_KERNEL])
    conv2 = conv2d(pool1, conv2_w)
    # 激活
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    # 池化
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 拉直
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 全连接一层
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 丢弃
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 全连接第二层
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b

    return y