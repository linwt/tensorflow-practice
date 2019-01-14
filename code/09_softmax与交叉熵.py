import tensorflow as tf

# ------------------- 交叉熵实验 -------------------
# 标签和网络输出值
labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

# 将logits分别进行1次和2次softmax
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

# 将以上两个值分别进行softmax_cross_entropy_with_logits
#正确的方式
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#如果将softmax变换完的值放进去，就相当于算第二次softmax的loss，所以会出错
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)

# 已进行过1次softmax，自定义交叉熵
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print('scaled=', sess.run((logits_scaled)))
    print('scaled2=', sess.run((logits_scaled2)))
    print('res1=', sess.run(result1))
    print('res2=', sess.run(result2))
    print('res3=', sess.run(result3))

#scaled= [[0.01791432 0.00399722 0.97808844]
# [0.04980332 0.04506391 0.90513283]]
# scaled2= [[0.21747023 0.21446465 0.56806517]
# [0.2300214  0.22893383 0.5410447 ]]
# res1= [0.02215516 3.0996735 ]             # 第一个是跟标签分类相符的，第二个是跟标签分类不符的
# res2= [0.56551915 1.4743223 ]
# res3= [0.02215518 3.0996735 ]             # 与res1一致


# ------------------- one_hot实验 -------------------
labels = [[0.4, 0.1, 0.5],[0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print('res4=', sess.run(result4))

# res4= [2.1721554 2.7696736]              # 正确和错误分类的交叉熵差别没有one_hot明显


# ------------------- sparse交叉熵的使用 -------------------
# 需要使用非one_hot标签。其实是0 1 2 三个类，等价001 010
labels = [2, 1]
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print('res5=', sess.run(result5))

# res5= [0.02215516 3.0996735 ]            # 与res1一致


# ------------------- 计算loss值 -------------------
# 对交叉熵后得到的数组取均值即为loss值
# 方式一：
#   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
# 方式二：
#   logits_scaled = tf.nn.softmax(logits)
#   loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_scaled), 1))
# 方式三：
#   logits_scaled = tf.nn.softmax(logits)
#   loss = -tf.reduce_sum(labels * tf.log(logits_scaled))

loss = tf.reduce_mean(result1)
with tf.Session() as sess:
    print('loss=', sess.run(loss))

# loss= 1.5609143