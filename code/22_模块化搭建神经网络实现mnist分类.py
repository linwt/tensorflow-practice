import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from PIL import Image
import numpy as np

# 前向传播
class mnist_forward(object):

    # 初始化参数
    def __init__(self):
        self.INPUT_NODE = 784
        self.OUTPUT_NODE = 10
        self.LAYER1_NODE = 500

    # 获取权重
    def get_weight(self, shape, regularizer):
        w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

    # 获取偏度
    def get_bias(self, shape):
        b = tf.Variable(tf.zeros(shape))
        return b

    # 前向传播
    def forward(self, x, regularizer):
        # 第一层
        w1 = self.get_weight([self.INPUT_NODE, self.LAYER1_NODE], regularizer)
        b1 = self.get_bias([self.LAYER1_NODE])
        y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        # 第二层
        w2 = self.get_weight([self.LAYER1_NODE, self.OUTPUT_NODE], regularizer)
        b2 = self.get_bias([self.OUTPUT_NODE])
        y = tf.matmul(y1, w2) + b2
        return y

# 反向传播
class mnist_backward(object):

    # 初始化参数
    def __init__(self):
        self.BATCH_SIZE = 200
        self.LEARNING_RATE_BASE = 0.1
        self.LEARNING_RATE_DECAY = 0.99
        self.REGULARIZER = 0.0001
        self.STEPS = 50000
        self.MOVING_AVERAGE_DECAY = 0.99
        self.MODEL_SAVE_PATH="log"
        self.MODEL_NAME="mnist_model"

    # 反向传播
    def backward(self, mnist):

        mf = mnist_forward()

        x = tf.placeholder(tf.float32, [None, mf.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mf.OUTPUT_NODE])
        y = mf.forward(x, self.REGULARIZER)
        global_step = tf.Variable(0, trainable=False)

        # 包含正则化的损失函数
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cem = tf.reduce_mean(ce)
        loss = cem + tf.add_n(tf.get_collection('losses'))

        # 退化学习率
        learning_rate = tf.train.exponential_decay(
            self.LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / self.BATCH_SIZE,
            self.LEARNING_RATE_DECAY,
            staircase=True)

        # 优化器
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 滑动平均
        ema = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        # 启动session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 实现断点续训，用模型中的W和b继续训练
            ckpt = tf.train.get_checkpoint_state(self.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for i in range(self.STEPS):
                xs, ys = mnist.train.next_batch(self.BATCH_SIZE)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 1000 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    saver.save(sess, os.path.join(self.MODEL_SAVE_PATH, self.MODEL_NAME), global_step=global_step)

    # After 1 training step(s), loss on training batch is 2.96259.
    # After 1001 training step(s), loss on training batch is 0.381561.
    # After 2001 training step(s), loss on training batch is 0.208591.
    # After 3001 training step(s), loss on training batch is 0.239505.
    # ...
    # After 46001 training step(s), loss on training batch is 0.13038.
    # After 47001 training step(s), loss on training batch is 0.124475.
    # After 48001 training step(s), loss on training batch is 0.126831.
    # After 49001 training step(s), loss on training batch is 0.123602.

# 测试准确率
class mnist_test(object):

    def __init__(self):
        self.TEST_INTERVAL_SECS = 5

    def test(self, mnist):

        mf = mnist_forward()
        mb = mnist_backward()

        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [None, mf.INPUT_NODE])
            y_ = tf.placeholder(tf.float32, [None, mf.OUTPUT_NODE])
            y = mf.forward(x, None)

            # 滑动平均
            ema = tf.train.ExponentialMovingAverage(mb.MOVING_AVERAGE_DECAY)
            ema_restore = ema.variables_to_restore()
            saver = tf.train.Saver(ema_restore)

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            while True:
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(mb.MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                        print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    else:
                        print('No checkpoint file found')
                        return
                time.sleep(self.TEST_INTERVAL_SECS)

    # After 3001 training step(s), test accuracy = 0.9667
    # After 3001 training step(s), test accuracy = 0.9667
    # After 4001 training step(s), test accuracy = 0.9709
    # After 4001 training step(s), test accuracy = 0.9709
    # After 5001 training step(s), test accuracy = 0.9737
    # ...
    # After 46001 training step(s), test accuracy = 0.9809
    # After 47001 training step(s), test accuracy = 0.9809
    # After 48001 training step(s), test accuracy = 0.9804
    # After 49001 training step(s), test accuracy = 0.981

# 判别自制图片
class mnist_app(object):

    # 加载模型，预测结果
    def restore_model(self, testPicArr):

        mf = mnist_forward()
        mb = mnist_backward()

        with tf.Graph().as_default() as tg:
            x = tf.placeholder(tf.float32, [None, mf.INPUT_NODE])
            y = mf.forward(x, None)
            preValue = tf.argmax(y, 1)

            variable_averages = tf.train.ExponentialMovingAverage(mb.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mb.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    preValue = sess.run(preValue, feed_dict={x:testPicArr})
                    return preValue
                else:
                    print("No checkpoint file found")
                    return -1

    # 预处理图片
    def pre_pic(self, picName):
        img = Image.open(picName)
        reIm = img.resize((28,28), Image.ANTIALIAS)
        # 图片转为灰度
        im_arr = np.array(reIm.convert('L'))
        threshold = 50
        # 黑白反转
        for i in range(28):
            for j in range(28):
                im_arr[i][j] = 255 - im_arr[i][j]
                if im_arr[i][j] < threshold:
                    im_arr[i][j] = 0
                else:
                    im_arr[i][j] = 255

        nm_arr = im_arr.reshape([1, 784])
        nm_arr = nm_arr.astype(np.float32)
        img_ready = np.multiply(nm_arr, 1.0/255.0)

        return img_ready

    def application(self):
        testNum = int(input("input the number of test pictures:"))
        for i in range(testNum):
            testPic = input("the path of test picture:")
            testPicArr = self.pre_pic(testPic)
            preValue = self.restore_model(testPicArr)
            print("The prediction number is:", preValue)

    # input the number of test pictures:2
    # the path of test picture:22_pic/3.png
    # The prediction number is: [3]
    # the path of test picture:22_pic/6.png
    # The prediction number is: [6]


def main():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # 运行反向传播代码后注释掉此代码，再运行测试准确率的代码，同步查看变化情况
    mb = mnist_backward()
    mb.backward(mnist)

    # mt = mnist_test()
    # mt.test(mnist)

    # app = mnist_app()
    # app.application()

if __name__ == '__main__':
    main()
