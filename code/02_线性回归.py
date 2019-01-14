import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

################ 以逻辑回归拟合二维数据 ################

# ----------- 数据准备 -----------

# 生成模拟数据，起始值为-1，结尾值为1，共100个数
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

# 显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()


# ----------- 创建模型 -----------

# 占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')

# 模型参数。tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 前向结构
z = tf.multiply(X, W) + b
# 将预测值以直方图形式显示
tf.summary.histogram('z', z)

# 反向传播
# 损失函数：均值平方差MSE
cost = tf.reduce_mean(tf.square(Y - z))
# 将损失值以标量形式显示
tf.summary.scalar('loss_function', cost)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# ----------- 训练模型 -----------

# 初始化所有变量
init = tf.global_variables_initializer()

# 定义参数
training_epochs = 20
display_step = 2

# 生成saver，迭代过程只保存一个文件
saver = tf.train.Saver(max_to_keep=1)
savedir = '02_log/'

# 存放批次值和损失值
plotdata = {'batchsize':[], 'loss':[]}

# 计算平均损失
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# 启动session
with tf.Session() as sess:
    sess.run(init)

    # 合并所有summary
    merged_summary_op = tf.summary.merge_all()
    # 创建summary_writer用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)

    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = {X: x, Y: y})

        # 生成summary
        summary_str = sess.run(merged_summary_op, feed_dict={X:x, Y:y})
        # 将summary写入文件
        summary_writer.add_summary(summary_str, epoch)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict = {X: train_X, Y: train_Y})
            print('Epoch:', epoch+1, 'cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))
            if not loss=='NA':
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)

    print('Finished!')

# ----------- 保存模型 -----------
    saver.save(sess, savedir + 'linermodel.cpkt', global_step=epoch)
    print('cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))

# ----------- 使用模型 -----------
    print('x=0.2, z=', sess.run(z, feed_dict={X: 0.2}))

    # Epoch: 1 cost= 0.522759 W= [0.9320547] b= [0.30152312]
    # Epoch: 3 cost= 0.10592335 W= [1.7020737] b= [0.09433816]
    # Epoch: 5 cost= 0.07542199 W= [1.9107836] b= [0.01573285]
    # Epoch: 7 cost= 0.074362166 W= [1.9649115] b= [-0.00501347]
    # Epoch: 9 cost= 0.074552424 W= [1.978909] b= [-0.01038455]
    # Epoch: 11 cost= 0.07463271 W= [1.9825287] b= [-0.01177359]
    # Epoch: 13 cost= 0.074655555 W= [1.9834651] b= [-0.01213287]
    # Epoch: 15 cost= 0.074661605 W= [1.9837071] b= [-0.0122257]
    # Epoch: 17 cost= 0.07466316 W= [1.9837692] b= [-0.01224962]
    # Epoch: 19 cost= 0.07466358 W= [1.983786] b= [-0.01225603]
    # Finished!
    # cost= 0.07466358 W= [1.9837887] b= [-0.01225712]
    # x=0.2, z= [0.38450062]


# ----------- 训练模型可视化 -----------

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()

    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('Minnibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()


# ----------- 载入检查点 -----------

# 指定检查点版本
load_epoch = 19
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    # 载入模型
    saver.restore(sess2, savedir + 'linermodel.cpkt-' + str(load_epoch))
    # 使用模型
    print('x=0.2, z=', sess2.run(z, feed_dict={X: 0.2}))

# 通过字典获取检查点文件
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(savedir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess3, ckpt.model_checkpoint_path)
        print('x=0.2, z=', sess3.run(z, feed_dict={X: 0.2}))

# 获取最后一个检查点文件
with tf.Session() as sess4:
    sess4.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(savedir)
    if ckpt:
        saver.restore(sess4, ckpt)
        print('x=0.2, z=', sess4.run(z, feed_dict={X: 0.2}))


# ----------- 查看模型内容 -----------

print_tensors_in_checkpoint_file(savedir + 'linermodel.cpkt-' + str(load_epoch), None, True)

# tensor_name:  bias
# [-0.01225712]
# tensor_name:  weight
# [1.9837887]


# 直接存储参数方式保存模型
WW = tf.Variable(1.0, name = 'WW')
bb = tf.Variable(2.0, name = 'bb')

saver = tf.train.Saver({'weight':WW, 'bias':bb})
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, savedir + 'linermodel.cpkt')
print_tensors_in_checkpoint_file(savedir + 'linermodel.cpkt', None, True)

# tensor_name:  bias
# 2.0
# tensor_name:  weight
# 1.0


# ----------- TensorBoard可视化 -----------

# 1、cmd中进入summary日志的上级路径下
# 2、输入tensorboard --logdir mnist_with_summaries(绝对路径)
# 3、Chrome浏览器输入 localhost:6006