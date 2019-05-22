import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import mnist_generateds

TEST_INTERVAL_SEC = 5
# 第六课，给出测试样本数
TEST_NUM = 10000

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 查看是否匹配
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 第六课， 给出样本
        # img_batch, label_batch = mnist_generateds.get_tfRecord(TEST_NUM, isTrain=False)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    """开启协调器，多线程"""
                    # coord = tf.train.Coordinator()
                    # threads = tf.train.start_queue_runners(sess, coord)

                    # xs, ys = sess.run(img_batch, label_batch)
                    # accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_: ys})

                    accuracy_score = sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels})
                    print("After %s training steps, test accuracy = %g" % (global_step, accuracy_score))

                    y_pred = sess.run(y, feed_dict={x:mnist.test.images, y_: mnist.test.labels})
                    return y_pred

                    # 关闭协调器
                    # coord.request_stop()
                    # coord.join(threads)

                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SEC)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    y_pred = test(mnist)
    return y_pred

if __name__ == '__main__':
    y_pred = main()
    print(y_pred)
