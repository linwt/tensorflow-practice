import tensorflow as tf

# Variable和get_variable的区别
var1 = tf.Variable(1.0, name='firstvar')
print('var1:', var1.name)
var1 = tf.Variable(2.0, name='firstvar')
print('var1:', var1.name)
var2 = tf.Variable(3.0)
print('var2:', var2.name)
var2 = tf.Variable(4.0)
print('var2:', var2.name)

get_var1 = tf.get_variable('firstvar', [1], initializer=tf.constant_initializer(5.0))
print('get_var1:', get_var1.name)
get_var1 = tf.get_variable('firstvar1', [1], initializer=tf.constant_initializer(6.0))
print('get_var1:', get_var1.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('var1=', var1.eval())
    print('var2=', var2.eval())
    print('get_var1=', get_var1.eval())

# var1: firstvar:0
# var1: firstvar_1:0
# var2: Variable:0
# var2: Variable_1:0
# get_var1: firstvar_2:0
# get_var1: firstvar1:0
# var1= 2.0
# var2= 4.0
# get_var1= [6.]

# 结论：
# 1、若名字重复，则添加序号区分。
# 2、若没有指定名字，则默认为Variable。
# 3、若变量重复，则最后一个的值生效。
# 4、get_variable只能定义一次指定名称的变量



# 特定作用域下获取变量
with tf.variable_scope('test1'):
    var1 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
    with tf.variable_scope('test2'):
        var2 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
print('var1:', var1.name)
print('var2:', var2.name)

# var1: test1/firstvar:0
# var2: test1/test2/firstvar:0



# 共享变量
with tf.variable_scope('test1', reuse=True):
    var3 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
    with tf.variable_scope('test2', reuse=True):
        var4 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
print('var3:', var3.name)
print('var4:', var4.name)

# var3: test1/firstvar:0
# var4: test1/test2/firstvar:0



# 初始化共享变量的作用域
with tf.variable_scope('test1', initializer=tf.constant_initializer(1.0)):
    var1 = tf.get_variable('firstval', shape=[2], dtype=tf.float32)
    with tf.variable_scope('test2'):
        var2 = tf.get_variable('firstval', shape=[2], dtype=tf.float32)
        var3 = tf.get_variable('var3', shape=[2], initializer=tf.constant_initializer(2.0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('var1:', var1.eval())
    print('var2:', var2.eval())
    print('var3:', var3.eval())

# var1: [1. 1.]
# var2: [1. 1.]
# var3: [2. 2.]



# 作用域与操作符的受限范围
with tf.variable_scope('scope1') as sp1:
    var1 = tf.get_variable('v', [1])
print('sp1:',sp1.name)
print('var1:',var1.name)
# sp1: scope1
# var1: scope1/v:0

with tf.variable_scope('scope2'):
    var2 = tf.get_variable('v', [1])

    with tf.variable_scope(sp1) as sp3:
        var3 = tf.get_variable('v3', [1])

        with tf.variable_scope('') :
            var4 = tf.get_variable('v4', [1])
print('sp3:',sp3.name)
print('var2:',var2.name)
print('var3:',var3.name)
print('var4:',var4.name)
# sp3: scope1
# var2: scope2/v:0
# var3: scope1/v3:0
# var4: scope1//v4:0

with tf.variable_scope('scope'):
    with tf.name_scope('bar'):
        v = tf.get_variable('v', [1])
        x = 1.0 + v
        with tf.name_scope(''):
            y = 1.0 + v
print('v:',v.name)
print('x.op:',x.op.name)
print('y.op:',y.op.name)
# v: scope/v:0
# x.op: scope/bar/add
# y.op: add

# 结论：
# 1、用as的方式定义作用域，作用域变量将不再受外围的scope限制
# 2、tf.name_scope只能限制op，不能限制变量
# 3、添加空字符作用域将返回到顶层