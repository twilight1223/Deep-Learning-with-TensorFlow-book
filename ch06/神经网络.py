# __author__:zsshi
# __date__:2019/11/27

import tensorflow as tf
"""
期初的神经网络是线性模型，不能用于线性不可分问题，加激活函数，将线性模型变为非线性
"""
x = tf.random.uniform([2,4])
w = tf.random.normal([4,2])
b = tf.zeros([2])
z1 = tf.nn.relu(x@w+b)
print(z1)

fc = tf.keras.layers.Dense(2,activation=tf.nn.relu)#自动初始化w和b
z2 = fc(x)
print(z2)
print(fc.kernel)
print(fc.bias)
print(fc.trainable_variables)
print(fc.variables)

"""
按层方式实现256，128, 64，10节点的网络连接
"""
fc1 = tf.keras.layers.Sequential([
    tf.keras.layers.Dense(256,tf.nn.relu),
    tf.keras.layers.Dense(128,tf.nn.relu),
    tf.keras.layers.Dense(64,tf.nn.relu),
    tf.keras.layers.Dense(10,tf.nn.softmax)
])
model = fc1(x)



