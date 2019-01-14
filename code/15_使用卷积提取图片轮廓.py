import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

myimg = mpimg.imread('img.png')
plt.imshow(myimg)
plt.axis('off')
plt.show()
print(myimg.shape)
# (高，宽，通道数)
# (650, 1300, 4)

full = np.reshape(myimg, [1, 650, 1300, 4])
inputfull = tf.Variable(tf.constant(1.0, shape=[1, 650, 1300, 4]))

# 卷积核个数1，高3宽3，由于通道数为4，则每个元素扩充成4个
filter = tf.Variable(tf.constant([[-1.0, -1.0, -1.0, -1.0], [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0],
                                  [-2.0, -2.0, -2.0, -2.0], [0, 0, 0, 0], [2.0, 2.0, 2.0, 2.0],
                                  [-1.0, -1.0, -1.0, -1.0], [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]], shape=[3, 3, 4, 1]))

op = tf.nn.conv2d(inputfull, filter, strides=[1, 1, 1, 1], padding='SAME')
# 归一化处理
o = tf.cast( ((op - tf.reduce_min(op)) / (tf.reduce_max(op) - tf.reduce_min(op)) ) * 255, tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t, f = sess.run([o, filter], feed_dict={inputfull:full})
    t = np.reshape(t, [650, 1300])

    plt.imshow(t, cmap='Greys_r')
    plt.axis('off')
    plt.show()