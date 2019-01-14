from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 导入图片数据集
mnist = input_data.read_data_sets('MNIST_data')
tf.reset_default_graph()

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None])

# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# maxout网络构建方法：通过reduce_max函数对多个神经元的输出来计算Max值，将Max值当做输入按照神经元正反传播方式进行计算
# 定义输出结点
z = tf.matmul(x, W) + b
maxout = tf.reduce_max(z, axis=1, keep_dims=True)
W2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]))
pred = tf.nn.softmax(tf.matmul(maxout, W2) + b2)

# 定义反向传播结构
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))
learning_rate = 0.04
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 定义训练参数
training_epochs = 25
batch_size = 100
display_step = 1

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print('Epoch:', epoch+1, 'cost=', avg_cost)

    print( " Finished!")

# Epoch: 0001 cost= 4.332749380
# Epoch: 0002 cost= 1.703540799
# Epoch: 0003 cost= 1.270259099
# Epoch: 0004 cost= 1.078033910
# Epoch: 0005 cost= 0.964661249
# Epoch: 0006 cost= 0.887960400
# Epoch: 0007 cost= 0.831142489
# Epoch: 0008 cost= 0.786481725
# Epoch: 0009 cost= 0.751113741
# Epoch: 0010 cost= 0.721779852
# Epoch: 0011 cost= 0.696367919
# Epoch: 0012 cost= 0.674368752
# Epoch: 0013 cost= 0.655132663
# Epoch: 0014 cost= 0.638249740
# Epoch: 0015 cost= 0.622899811
# Epoch: 0016 cost= 0.609075351
# Epoch: 0017 cost= 0.596355237
# Epoch: 0018 cost= 0.585118527
# Epoch: 0019 cost= 0.574638296
# Epoch: 0020 cost= 0.564739034
# Epoch: 0021 cost= 0.555896843
# Epoch: 0022 cost= 0.547583903
# Epoch: 0023 cost= 0.539620938
# Epoch: 0024 cost= 0.532442445
# Epoch: 0025 cost= 0.525198853
# Finished!