import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义网络模型参数
n_input = 784
n_hidden1 = 256
n_hidden2 = 256
n_classes = 10

# 定义占位符
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

# 定义学习参数
weight = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 正向结构
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight['h1']), biases['b1']))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weight['h2']), biases['b2']))
pred = tf.matmul(layer_2, weight['out']) + biases['out']

# 反向传播
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 定义参数
training_epoch = 25
batch_size = 100
display_step = 1

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 启动循环开始训练
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for _ in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})
            # 计算平均loss值
            avg_cost += c/total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print('Epoch:', epoch+1, 'cost=', avg_cost)

    print('Finished!')

# Epoch: 1 cost= 47.486816027814655
# Epoch: 2 cost= 8.730623123367183
# Epoch: 3 cost= 4.819391269151712
# Epoch: 4 cost= 3.302575463351744
# Epoch: 5 cost= 2.5068996660592373
# Epoch: 6 cost= 2.087587423069197
# Epoch: 7 cost= 1.748706776138417
# Epoch: 8 cost= 1.7081841864611835
# Epoch: 9 cost= 1.3306108449791807
# Epoch: 10 cost= 1.4699138337906954
# Epoch: 11 cost= 1.1788998931664725
# Epoch: 12 cost= 1.0357377399670027
# Epoch: 13 cost= 0.95466765865809
# Epoch: 14 cost= 0.8442869940551242
# Epoch: 15 cost= 0.7171895869208853
# Epoch: 16 cost= 0.8400323700072798
# Epoch: 17 cost= 0.5354551338023037
# Epoch: 18 cost= 0.5231031399928924
# Epoch: 19 cost= 0.43841067512285536
# Epoch: 20 cost= 0.5062140923094006
# Epoch: 21 cost= 0.43025539846236704
# Epoch: 22 cost= 0.33604809949580366
# Epoch: 23 cost= 0.2797555153112193
# Epoch: 24 cost= 0.26059670231142323
# Epoch: 25 cost= 0.292627252683084
# Finished!
