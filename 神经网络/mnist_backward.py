import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import mnist_generateds

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

"""第六课"""
# 多线程
train_num_examples = 60000  # 代替mnist.train.num_examples


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    """Computes sparse softmax cross entropy between `logits` and `labels`."""
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        # train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')

    saver = tf.train.Saver()
    # 生成数据 """第六课"""
    # img_batch, laber_batch = mnist_generateds.generate_tfRecord(BATCH_SIZE, isTrained = True)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        """开启协调器，多线程"""
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess, coord)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 第六课， 获取数据
            # xs, ys = sess.run([img_batch, laber_batch])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
            if i%1000 == 0:
                print("After %d training steps, loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        # 关闭协调器
        # coord.request_stop()
        # coord.join(threads)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)
    # 第六课
    # backward()

if __name__ == '__main__':
    main()