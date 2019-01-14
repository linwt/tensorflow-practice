import tensorflow as tf

# 常数
A = tf.constant(3)
B = tf.constant(4)
with tf.Session() as sess:
    print('相加：%i' % sess.run(A+B))
    print('相乘：%i' % sess.run(A*B))



# 占位符，注入机制
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    print('相加：%i' % sess.run(add, feed_dict={a:3, b:4}))
    print('相乘：%i' % sess.run(mul, feed_dict={a:3, b:4}))
    print(sess.run([add, mul], feed_dict={a:3, b:4}))