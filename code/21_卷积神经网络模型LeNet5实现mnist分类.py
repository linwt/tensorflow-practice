import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

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

with tf.variable_scope('dropout'):
    dropout_output = tf.nn.dropout(connection_output, 0.7)
    
with tf.name_scope('Output'):
    weight = tf.Variable(tf.truncated_normal([80, 10]),dtype= tf.float32)
    bias = tf.Variable(tf.truncated_normal([10]),dtype= tf.float32)
    y_model = tf.nn.softmax(tf.add(tf.matmul(dropout_output, weight), bias))

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_data, logits = y_model))
    tf.summary.scalar('The variation of the loss', loss)

with tf.name_scope('Accuracy'):
    prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    tf.summary.scalar('The variation of the accuracy', accuracy)

with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("21_log/", sess.graph)
    merged = tf.summary.merge_all()
    a = []
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x_data: batch_x, y_data: batch_y})
        print('epoch:', epoch, ',accuracy:', sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
        summary, acc = sess.run([merged, accuracy], feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels})
        a.append(acc)
        writer.add_summary(summary, epoch)
    writer.close()

    # 绘制训练精度变化图
    plt.plot(a)
    plt.title('The variation of the acuracy')
    plt.xlabel('The sampling point')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

# epoch: 0 ,accuracy: 0.2409
# epoch: 1 ,accuracy: 0.3678
# epoch: 2 ,accuracy: 0.4367
# epoch: 3 ,accuracy: 0.5451
# epoch: 4 ,accuracy: 0.6318
# epoch: 5 ,accuracy: 0.6978
# epoch: 6 ,accuracy: 0.7291
# epoch: 7 ,accuracy: 0.7388
# epoch: 8 ,accuracy: 0.7763
# epoch: 9 ,accuracy: 0.7666
# epoch: 10 ,accuracy: 0.7734
# epoch: 11 ,accuracy: 0.7958
# epoch: 12 ,accuracy: 0.8025
# epoch: 13 ,accuracy: 0.8005
# epoch: 14 ,accuracy: 0.8065
# epoch: 15 ,accuracy: 0.8007
# epoch: 16 ,accuracy: 0.8091
# epoch: 17 ,accuracy: 0.8176
# epoch: 18 ,accuracy: 0.811
# epoch: 19 ,accuracy: 0.8212
# epoch: 20 ,accuracy: 0.8183
