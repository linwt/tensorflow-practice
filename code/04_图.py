import numpy as np
import tensorflow as tf

# 建立图
c = tf.constant(0.0)
g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)
    c2 = tf.constant(1.0)
    print(c1.graph)
    print(g)
    print(c.graph)

g2 = tf.get_default_graph()
print(g2)

tf.reset_default_graph()
g3 = tf.get_default_graph()
print(g3)

# <tensorflow.python.framework.ops.Graph object at 0x00000181F1096E48>
# <tensorflow.python.framework.ops.Graph object at 0x00000181F1096E48>
# <tensorflow.python.framework.ops.Graph object at 0x00000181E9BC9748>
# <tensorflow.python.framework.ops.Graph object at 0x00000181E9BC9748>
# <tensorflow.python.framework.ops.Graph object at 0x00000181F10AC128>



# 获取张量
print(c1.name)
t = g.get_tensor_by_name(name='Const:0')
print(t)

# Const:0
# Tensor("Const:0", shape=(), dtype=float32)



# 获取元素列表
t2 = g.get_operations()
print(t2)

# [<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>]



# 获取对象。输入一个对象，返回一个张量或OP
t3 = g.as_graph_element(c1)
print(t3)

# Tensor("Const:0", shape=(), dtype=float32)