import tensorflow as tf

# 定义输入变量，4x4大小的2通道矩阵
img = tf.constant([
    [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
    [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
    [[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
    [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]]
])
img = tf.reshape(img, [1, 4, 4, 2])

# tf.nn.max_pool(input, ksize, strides, padding, name=None)
# tf.nn.avg_pool(input, ksize, strides, padding, name=None)
# 定义池化操作
pooling = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
pooling1 = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')
pooling2 = tf.nn.avg_pool(img, [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')
pooling3 = tf.nn.avg_pool(img, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')
nt_hpool2_flat = tf.reshape(tf.transpose(img), [-1, 16])
pooling4 = tf.reduce_mean(nt_hpool2_flat, 1)   # 1：对行求均值  0：对列求均值

# 运行池化操作
with tf.Session() as sess:
    image = sess.run(img)
    print("image:\n", image)

    result = sess.run(pooling)
    print("reslut:\n", result)

    result1 = sess.run(pooling1)
    print("reslut1:\n", result1)

    result2 = sess.run(pooling2)
    print("reslut2:\n", result2)

    result3 = sess.run(pooling3)
    print("reslut3:\n", result3)

    flat, result4 = sess.run([nt_hpool2_flat, pooling4])
    print("reslut4:\n", result4)
    print("flat:\n", flat)

# image:
# [[[[0. 4.]
#    [0. 4.]
#   [0. 4.]
#  [0. 4.]]
#
# [[1. 5.]
#  [1. 5.]
# [1. 5.]
# [1. 5.]]
#
# [[2. 6.]
#  [2. 6.]
# [2. 6.]
# [2. 6.]]
#
# [[3. 7.]
#  [3. 7.]
# [3. 7.]
# [3. 7.]]]]

# reslut:
# [[[[1. 5.]
#    [1. 5.]]
#
#  [[3. 7.]
#     [3. 7.]]]]

# reslut1:
# [[[[1. 5.]
#    [1. 5.]
#   [1. 5.]]
#
# [[2. 6.]
# [2. 6.]
# [2. 6.]]
#
# [[3. 7.]
#  [3. 7.]
# [3. 7.]]]]

# reslut2:
# [[[[1.  5. ]
#    [1.  5. ]
#   [1.  5. ]
#  [1.  5. ]]
#
# [[1.5 5.5]
#  [1.5 5.5]
# [1.5 5.5]
# [1.5 5.5]]
#
# [[2.  6. ]
#  [2.  6. ]
# [2.  6. ]
# [2.  6. ]]
#
# [[2.5 6.5]
#  [2.5 6.5]
# [2.5 6.5]
# [2.5 6.5]]]]

# reslut3:
# [[[[1.5 5.5]]]]

# reslut4:
# [1.5 5.5]
# flat:
# [[0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
#  [4. 5. 6. 7. 4. 5. 6. 7. 4. 5. 6. 7. 4. 5. 6. 7.]]