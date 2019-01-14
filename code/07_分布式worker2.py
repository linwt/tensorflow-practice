import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()

#定义ip和端口
strps_hosts="localhost:1681"
strworker_hosts="localhost:1682,localhost:1683"

#定义角色名称
strjob_name = "worker"
task_index = 1

# 将字符串转成数组
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})

#创建server
server = tf.train.Server(
    {'ps': ps_hosts,'worker': worker_hosts},
    job_name=strjob_name,
    task_index=task_index)

#ps角色使用join进行等待
if strjob_name == 'ps':
    print("wait")
    server.join()

#创建网络结构
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster_spec)):

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # 模型参数
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")

    #获得迭代次数
    global_step = tf.contrib.framework.get_or_create_global_step()

    # 前向结构
    z = tf.multiply(X, W)+ b
    #将预测值以直方图显示
    tf.summary.histogram('z',z)

    #反向优化
    cost =tf.reduce_mean(tf.square(Y - z))
    #将损失以标量显示
    tf.summary.scalar('loss_function', cost)
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)
    #合并所有summary
    merged_summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()


#创建Supervisor，管理session

#定义参数
training_epochs = 2200
display_step = 2

sv = tf.train.Supervisor(is_chief=(task_index == 0),
                         logdir="log/super/",
                         init_op=init,
                         summary_op=None,
                         saver=saver,
                         global_step=global_step,
                         save_model_secs=5)

#连接目标角色创建session
with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session=sess))

    for epoch in range(global_step.eval(session=sess),training_epochs*len(train_X)):
        for (x, y) in zip(train_X, train_Y):
            _, epoch = sess.run([optimizer,global_step] ,feed_dict={X: x, Y: y})
            #生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X: x, Y: y})
            #将summary 写入文件。worker2不是chief supervisor，不能进行写入
            #sv.summary_computed(sess, summary_str,global_step=epoch)
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
                if not (loss == "NA" ):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)

    print (" Finished!")
    sv.saver.save(sess,"log/mnist_with_summaries/"+"sv.cpk",global_step=epoch)

sv.stop()