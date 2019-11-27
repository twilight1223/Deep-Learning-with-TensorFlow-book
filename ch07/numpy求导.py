# __author__:zsshi
# __date__:2019/11/27

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# x = np.linspace(-5,5,10)
# # sigmoid激活函数
#
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# def derivative(x):
#     return sigmoid(x)*(1-sigmoid(x))
#

# plt.plot(x,sigmoid(x),label='sigmoid')
# plt.plot(x,derivative(x),label='d_sigmoid')
# plt.legend()
# plt.show()

# relu
# def relu(x):
#     d = np.array(x,copy=True)
#     d[x<0] = 0
#     d[x>0] = x[x>0]
#     return d
#
# def d_relu(x):
#     d = np.array(x,copy=True)
#     d[x<0] = 0
#     d[x>0] = 1
#     return d
#
# plt.plot(x,relu(x),label='relu')
# plt.plot(x,d_relu(x),label='d_relu')
# plt.show()

# leakey_relu函数求导
# def leakey_relu(x):
#     d = np.array(x,copy=True)
#     d[x>=0] = x[x>=0]
#     d[x<0] = x[x<0]*0.01
#     return d
# def d_leakey_relu(x):
#     d = np.array(x,copy=True)
#     d[x>=0] = 1
#     d[x<0] = 0.01
#     return d
# plt.plot(x,leakey_relu(x),label='leakey_relu')
# plt.plot(x,d_leakey_relu(x),label='d_leakey_relu')
# plt.show()

# tanh函数梯度
# def tanh(x):
#     return 2*sigmoid(2*x)-1
#
# # d_tanh
# def d_tanh(x):
#     return 1-tanh(x)**2
#
# plt.plot(x,tanh(x),label='tanh')
# plt.plot(x,d_tanh(x),label='d_tanh')
# plt.show()

# mse 梯度 o-y
# 交叉熵梯度 p-y


# 全连接网络求导
x = tf.random.normal([2,4],dtype=tf.float32)
w1 = tf.Variable(tf.random.normal([4,10]))
b1 = tf.Variable(tf.zeros([10]))
w2 = tf.Variable(tf.random.normal([10,1]))
b2 = tf.Variable(tf.zeros([1]))

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1,b1,w2,b2])
    y1 = x@w1+b1
    y2 = y1@w2+b2

dy2_dy1 = tape.gradient(y2,[y1])[0]
dy1_dw1 = tape.gradient(y1,[w1])[0]
dy2_dw1 = tape.gradient(y2,[w1])[0]

print(dy2_dw1)
l1 = tf.concat([dy2_dy1,dy2_dy1],axis=0) #多节点运算，需要矩阵拼接
print(tf.multiply(l1,dy1_dw1))
# tf.multiply









