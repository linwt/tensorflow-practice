import tensorflow as tf
import numpy as np

# 使用带隐藏层的神经网络拟合异或操作
# 网络结构：2维输入 → 2维隐藏层 → 1维输出

# 定义变量
learning_rate = 1e-4
n_input = 2
n_hidden = 2
n_label = 1

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])

# 定义学习参数
weight = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
}
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label])),
}

# 定义网络模型
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight['h1']), biases['h1']))
y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2']))
# y_pred = tf.nn.relu(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2']))
# y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight['h2']), biases['h2']))

loss = tf.reduce_mean((y_pred - y) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 生成数据
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')

# 加载session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练
for _ in range(10000):
    sess.run(optimizer, feed_dict={x:X, y:Y})

# 计算预测值
print(sess.run(y_pred, feed_dict={x:X}))
# [[0.11455312]     →  0
#  [0.5669606 ]     →  1
#  [0.7539438 ]     →  1
#  [0.3307021 ]]    →  0

# 查看隐藏层的输出
print(sess.run(layer_1, feed_dict={x:X}))
# [[0.         0.        ]
#  [0.72017604 0.        ]
# [0.         0.85631204]
# [0.31176186 0.        ]]