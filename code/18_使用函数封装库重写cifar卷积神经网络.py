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
# 定义占位符
x = tf.placeholder(tf.float32, [None, 24,24,3])  # cifar图片形状24*24*3
y = tf.placeholder(tf.float32, [None, 10])       # 0-9共10个类别

# 定义正向传播结构
x_image = tf.reshape(x, [-1, 24, 24, 3])
# 卷积层
h_conv1 = tf.contrib.layers.conv2d(x_image, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
# 池化层
h_pool1 = tf.contrib.layers.max_pool2d(h_conv1, [2, 2], stride=2, padding='SAME')

h_conv2 = tf.contrib.layers.conv2d(h_pool1, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
h_pool2 = tf.contrib.layers.max_pool2d(h_conv2, [2, 2], stride=2, padding='SAME')

# 全局平均池化
nt_hpool2 = tf.contrib.layers.avg_pool2d(h_pool2, [6, 6], stride=6, padding='SAME')

# 全连接层
nt_hpool2_flat = tf.reshape(nt_hpool2, [-1, 64])
y_conv = tf.contrib.layers.fully_connected(nt_hpool2_flat, 10, activation_fn=tf.nn.softmax)

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