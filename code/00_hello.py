import tensorflow as tf

# 定义一个常量
hello = tf.constant('Hello, TensorFlow!')
# 建立一个session
sess = tf.Session()
# 通过session里面的run来运行结果
print(sess.run(hello))
# 关闭session
sess.close()