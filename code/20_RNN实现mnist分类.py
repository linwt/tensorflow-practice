import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入 MINST 数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义参数
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

tf.reset_default_graph()

# 定义占位符
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 对矩阵进行分解
x1 = tf.unstack(x, n_steps, 1)

# 1、单层LSTM
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

# 2、单层LSTM
# lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
# outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

# 3、单层GRU
# gru = tf.contrib.rnn.GRUCell(n_hidden)
# outputs = tf.contrib.rnn.static_rnn(gru, x1, dtype=tf.float32)

# 4、动态单层RNN
# gru = tf.contrib.rnn.GRUCell(n_hidden)
# outputs, _ = tf.nn.dynamic_rnn(gru, x, dtype=tf.float32)
# outputs = tf.transpose(outputs, [1, 0, 2])

# 5、静态多层LSTM
# stacked_rnn = []
# for i in range(3):
#     stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
# mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
# outputs, states = tf.contrib.rnn.staic_rnn(mcell, x1, dtype=tf.float32)

# 6、静态多层LSTM连接GRU
# gru = tf.contrib.rnn.GRUCell(n_hidden*2)
# lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
# mcell = tf.contrib.rnn.MultiRNNCell([lstm_cell, gru])
# outputs, states = tf.contrib.rnn.staic_rnn(mcell, x1, dtype=tf.float32)

# 7、动态多层RNN
# gru = tf.contrib.rnn.GRUCell(n_hidden*2)
# lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
# mcell = tf.contrib.rnn.MultiRNNCell([lstm_cell, gru])
# outputs, states = tf.contrib.rnn.dynamic_rnn(mcell, x1, dtype=tf.float32)
# outputs = tf.transpose(outputs, [1, 0, 2])

# 8、单层动态双向RNN
# lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
# print(len(outputs), outputs[0].shape, outputs[1].shape)
# outputs = tf.concat(outputs, 2)
# outputs = tf.transpose(outputs, [1, 0, 2])

# 9、单层静态双向RNN
# lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x1, dtype=tf.float32)

# 10、多层双向RNN
# lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn([lstm_fw_cell], [lstm_bw_cell], x1, dtype=tf.float32)

# 11、Multi双向RNN
# stacked_rnn = []
# stacked_bw_rnn = []
# for i in range(3):
#     stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
#     stacked_bw_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
# mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
# mcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn)
# outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn([mcell], [mcell_bw], x1, dtype=tf.float32)

# 12、动态多层双向RNN
# stacked_rnn = []
# stacked_bw_rnn = []
# for i in range(3):
#     stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
#     stacked_bw_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
# mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
# mcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn)
# outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([mcell], [mcell_bw], x1, dtype=tf.float32)
# outputs = tf.transpose(outputs, [1, 0, 2])

# 全连接层
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

# 定义模型参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估模型
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 重塑数据为28个 含有28个元素的 序列
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # 运行优化器
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # 计算损失值
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step*batch_size) + \
                   ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                   ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print (" Finished!")

    # 使用测试集计算准确率
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

# Iter 1280, Minibatch Loss= 2.145514, Training Accuracy= 0.25000
# Iter 2560, Minibatch Loss= 1.943221, Training Accuracy= 0.28125
# Iter 3840, Minibatch Loss= 1.704110, Training Accuracy= 0.38281
# Iter 5120, Minibatch Loss= 1.231408, Training Accuracy= 0.58594
# Iter 6400, Minibatch Loss= 1.210194, Training Accuracy= 0.55469
# ......
# Iter 96000, Minibatch Loss= 0.067738, Training Accuracy= 0.96875
# Iter 97280, Minibatch Loss= 0.119918, Training Accuracy= 0.94531
# Iter 98560, Minibatch Loss= 0.046925, Training Accuracy= 0.98438
# Iter 99840, Minibatch Loss= 0.155072, Training Accuracy= 0.95312
# Finished!
# Testing Accuracy: 0.984375