'''
    基本的 tensorflow 框架，实现了 y = x*0.1 + 0.3 参数的拟合。
'''


import tensorflow as tf
import numpy as np


# 生成100个[0,1]数字的列表，
# tensorflow 中的数字类型为float32，
# 所以我们这里将生成的随机数转换为这种数据类型。
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# 建立 tensorflow 的基本结构 start #
# 变量 一维的结构
# 首先定义 weights 表示0.1，初始值为 -0.1~0.1
# 其次定义 biases 表示0.3，初始值为 0
# 经过训练能够使他接近于给定参数值
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# 预测值和参数值之间的差别
# 并且根据给定的学习效率
# 通过训练，将误差值减小
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# 建立 tensorflow 的基本结构 end #



