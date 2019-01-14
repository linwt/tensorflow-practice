# 引入批量归一化：最大限度地保证每次的正向传播在同一分布上，这样反向计算时参照的数据样本分布就会与正向计算时的数据分布一样了。
# 将每一层运算出来的数据都归一化成均值为0方差为1 的标准高斯分布。这样就会在保留样本分布特征的同时，又消除了层与层间的分布差异。

import cifar10_input
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm

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

def batch_norm_layer(value, train=None, name='batch_norm'):
    if train is not None:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)
    else:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=False)

# 定义占位符
x = tf.placeholder(tf.float32, [None, 24,24,3])  # cifar图片形状24*24*3
y = tf.placeholder(tf.float32, [None, 10])       # 0-9共10个类别
train = tf.placeholder(tf.float32)

# 定义正向传播结构

# 获取权重
W_conv1 = weight_variable([5, 5, 3, 64])
# 获取偏度
b_conv1 = bias_variable([64])
# 卷积层
x_image = tf.reshape(x, [-1, 24, 24, 3])
h_conv1 = tf.nn.relu(batch_norm_layer(conv2d(x_image, W_conv1) + b_conv1, train))
# 池化层
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2) + b_conv2, train))
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
nt_hpool3 = avg_pool_6x6(h_conv3)

nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)

# 定义反向传播结构
# 定义退化学习率
global_step = tf.Variable(0, trainable=False)
decaylearning_rate = tf.train.exponential_decay(0.04, global_step, 1000, 0.9)
# 定义损失函数
loss = -tf.reduce_sum(y*tf.log(y_conv))
# 定义优化器
optimizer = tf.train.AdamOptimizer(decaylearning_rate).minimize(loss, global_step=global_step)

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