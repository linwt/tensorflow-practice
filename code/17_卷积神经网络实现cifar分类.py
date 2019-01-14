# 建立一个带有全局平均池化层的卷积神经网络

import cifar10_input
import tensorflow as tf
import numpy as np

# ---------- 导入头文件引入数据集 ----------
batch_size = 128
data_dir = './tmp/cifar10_data/cifar-10-batches-bin'
print("begin")
# 默认使用测试数据集，eval_data=False使用训练数据集
images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
print("begin data")

# ---------- 定义网络结构 ----------
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')

# 定义占位符
x = tf.placeholder(tf.float32, [None, 24,24,3])  # cifar图片形状24*24*3
y = tf.placeholder(tf.float32, [None, 10])       # 0-9共10个类别

# 定义正向传播结构

x_image = tf.reshape(x, [-1, 24, 24, 3])
# 获取权重
W_conv1 = weight_variable([5, 5, 3, 64])
# 获取偏度
b_conv1 = bias_variable([64])
# 卷积层
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化层
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
nt_hpool3 = avg_pool_6x6(h_conv3)

nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)

# 定义反向传播结构
loss = -tf.reduce_sum(y*tf.log(y_conv))
learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# ---------- 运行session进行训练 ----------
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()
for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]

    optimizer.run(feed_dict={x: image_batch, y: label_b})

    if i%200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_b})
        print("step %d, training accuracy %g" % (i, train_accuracy))

# ---------- 评估结果 ----------
image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch]
print ("finished！ test accuracy %g" % accuracy.eval(feed_dict={x: image_batch, y: label_b}))

# step 0, training accuracy 0.15625
# step 200, training accuracy 0.3125
# step 400, training accuracy 0.35973
# step 600, training accuracy 0.3125
# ......
# step 14400, training accuracy 0.554688
# step 14600, training accuracy 0.601526
# step 14800, training accuracy 0.5625
# finished！ test accuracy 0.632812