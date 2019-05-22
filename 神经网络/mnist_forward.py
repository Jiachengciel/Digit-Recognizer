import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight(shape, regulizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regulizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regulizer)(w))

    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regulizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regulizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regulizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    return y