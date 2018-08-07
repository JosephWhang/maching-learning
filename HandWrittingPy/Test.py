from numpy import *
import matplotlib.pyplot as plt
import tensorflow as tf

group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
print(group[:,0],group[:,1])

a = 3**2
print(a)

x = tf.placeholder("float", shape=[None, 784])
print(x)

W = tf.Variable(tf.zeros([784, 10]))
print(W)