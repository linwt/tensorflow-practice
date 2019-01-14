import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

# ----------------- 导入图片数据集 ---------------------
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)    # 手动导入数据集
print('训练集：', mnist.train.images)
print('训练集shape：', mnist.train.images.shape)

im = mnist.train.images[1]
im = im.reshape(-1, 28)
pylab.imshow(im)
pylab.show()

print('测试集shape：', mnist.test.images.shape)
print('验证集shape：', mnist.validation.images.shape)

# Extracting E:\IntelliJIDEA_AllProject\tensorflow\MNIST_data\train-images-idx3-ubyte.gz
# Extracting E:\IntelliJIDEA_AllProject\tensorflow\MNIST_data\train-labels-idx1-ubyte.gz
# Extracting E:\IntelliJIDEA_AllProject\tensorflow\MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting E:\IntelliJIDEA_AllProject\tensorflow\MNIST_data\t10k-labels-idx1-ubyte.gz
# 训练集：
# [[0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# ...
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]]
# 训练集shape： (55000, 784)
# 测试集shape： (10000, 784)
# 验证集shape： (5000, 784)

# ----------------- 定义变量 ---------------------
tf.reset_default_graph()

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# ----------------- 构建模型 ---------------------
# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出结点。softmax分类
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义反向传播结构
# 损失函数使用交叉熵运算后取均值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# ----------------- 训练模型 ---------------------
# 定义训练参数
training_epoch = 25
batch_size = 100
display_step = 1

# 定义saver
saver = tf.train.Saver()
model_path = 'log/mnist_model.ckpt'

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

    # Epoch: 1 cost= 9.596292721141479
    # Epoch: 2 cost= 4.972864401990715
    # Epoch: 3 cost= 3.4004221237789483
    # Epoch: 4 cost= 2.6458645525845625
    # Epoch: 5 cost= 2.2065195288441406
    # Epoch: 6 cost= 1.9224881771477766
    # Epoch: 7 cost= 1.724120068983598
    # Epoch: 8 cost= 1.5767061001604237
    # Epoch: 9 cost= 1.4620138127153586
    # Epoch: 10 cost= 1.370203634175388
    # Epoch: 11 cost= 1.2947486288981
    # Epoch: 12 cost= 1.2314134088429531
    # Epoch: 13 cost= 1.1779150407964543
    # Epoch: 14 cost= 1.1317333636500622
    # Epoch: 15 cost= 1.0914437276666835
    # Epoch: 16 cost= 1.0559651392156417
    # Epoch: 17 cost= 1.0244605872847816
    # Epoch: 18 cost= 0.9961965002796879
    # Epoch: 19 cost= 0.9708386997201223
    # Epoch: 20 cost= 0.9476161807775488
    # Epoch: 21 cost= 0.9266105699539184
    # Epoch: 22 cost= 0.9072015644203527
    # Epoch: 23 cost= 0.889332883412187
    # Epoch: 24 cost= 0.872809675498443
    # Epoch: 25 cost= 0.8575061844695698
    # Finished!

# ----------------- 测试模型 ---------------------
    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    # Accuracy: 0.8282

# ----------------- 保存模型 ---------------------
    save_path = saver.save(sess, model_path)
    print('Model saved in file:', save_path)

    # Model saved in file: log/mnist_model.ckpt

# ----------------- 读取模型 ---------------------
print('Starting 2nd session...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 加载模型
    saver.restore(sess, model_path)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x:batch_xs})
    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    # Starting 2nd session...
    # Accuracy: 0.8282
    # [5 1]
    # [[4.4982876e-03 1.1288207e-12 1.8214343e-12 2.3222594e-02 1.4197039e-08
    #  9.7227716e-01 4.4992859e-13 2.1727722e-13 2.0280638e-06 2.0632596e-11]
    # [1.2847535e-07 9.9965501e-01 1.7479906e-04 6.2280596e-05 6.3944384e-08
    # 1.9698538e-08 1.9706006e-09 2.3255639e-10 1.0761410e-04 9.2783906e-11]]
    # [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    # [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]