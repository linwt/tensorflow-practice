import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.examples.tutorials.mnist import input_data

# 设置学习率
learning_rate = 0.01
# 设置训练次数
train_steps = 1000

# 定义卷积层
def conv(input, filter_shape, bias_shape, strides_shape):
    filter = tf.get_variable("filter", filter_shape, initializer= tf.truncated_normal_initializer())
    bias = tf.get_variable("bias", bias_shape, initializer= tf.truncated_normal_initializer())
    conv = tf.nn.conv2d(input, filter, strides= strides_shape, padding= 'SAME')
    output = tf.nn.sigmoid(conv + bias)
    return output

# 定义池化层
def pooling(input, ksize_shape, strides_shape):
    output = tf.nn.max_pool(input, ksize= ksize_shape, strides= strides_shape, padding = 'SAME')
    return output

# 定义全连接层
def connection(input, weight_shape, bias_shape, flat_shape):
    weight = tf.get_variable("weight", weight_shape, initializer= tf.truncated_normal_initializer())
    bias = tf.get_variable("bias", bias_shape, initializer= tf.truncated_normal_initializer())

    flat = tf.reshape(input, flat_shape)
    output = tf.nn.sigmoid(tf.matmul(flat, weight) + bias)
    return output

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('Input'):
    x_data = tf.placeholder(tf.float32, [None, 784])
    y_data = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x_data, [-1, 28, 28, 1])

with tf.variable_scope('Conv1'):
    conv1_output = conv(x_image, [5, 5, 1, 6], [6], [1, 1, 1, 1])

with tf.variable_scope('Pooling1'):
    pooling1_output = pooling(conv1_output, [1, 2, 2, 1], [1, 2, 2, 1])

with tf.variable_scope('Conv2'):
    conv2_output = conv(pooling1_output, [5, 5, 6, 16], [16], [1, 1, 1, 1])

with tf.variable_scope('Pooling2'):
    pooling2_output = pooling(conv2_output, [1, 2, 2, 1], [1, 2, 2, 1])

with tf.variable_scope('Conv3'):
    conv3_output = conv(pooling2_output, [5, 5, 16, 120], [120], [1, 1, 1, 1])

with tf.variable_scope('Connection'):
    connection_output = connection(conv3_output, [7*7*120, 80], [80], [-1, 7*7*120])

with tf.name_scope('Output'):
    weight = tf.Variable(tf.truncated_normal([80, 10]),dtype= tf.float32)
    bias = tf.Variable(tf.truncated_normal([10]),dtype= tf.float32)
    y_model = tf.nn.softmax(tf.add(tf.matmul(connection_output, weight), bias))

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_data, logits = y_model))
    tf.summary.scalar('The variation of the loss', loss)

with tf.name_scope('Accuracy'):
    prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    tf.summary.scalar('The variation of the accuracy', accuracy)

with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_batch, y_batch = mnist.train.next_batch(50)
    writer = tf.summary.FileWriter("21_log/", sess.graph)
    merged = tf.summary.merge_all()
    batch_x, batch_y = mnist.train.next_batch(200)
    a = []
    for _ in range(train_steps):
        sess.run(train_op, feed_dict={x_data: batch_x, y_data: batch_y})
        if _ % 50 == 0:
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
            summary, acc = sess.run([merged, accuracy], feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels})
            a.append(acc)
            writer.add_summary(summary, _)
    writer.close()

    # 绘制训练精度变化图
    plt.plot(a)
    plt.title('The variation of the acuracy')
    plt.xlabel('The sampling point')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

# 0.1421
# 0.101
# 0.101
# 0.0974
# 0.101
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974
# 0.0974