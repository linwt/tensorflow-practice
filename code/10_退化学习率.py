import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=10,
                                           decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(learning_rate))
    for _ in range(20):
        step, rate = sess.run([add_global, learning_rate])
        print(step, rate)

# 0.1
# 1 0.1
# 2 0.09791484
# 3 0.09688862
# 4 0.095873155
# 5 0.094868325
# 6 0.09387404
# 7 0.092890166
# 8 0.09191661
# 9 0.09095325
# 10 0.089999996
# 11 0.08905673
# 12 0.08812335
# 13 0.087199755
# 14 0.087199755
# 15 0.0853815
# 16 0.08448663
# 17 0.08360115
# 18 0.08272495
# 19 0.08185793
# 20 0.08099999