# __author__:zsshi
# __date__:2019/11/27

import tensorflow as tf

# 实数输出，输出层不加激活函数

# softmax 输出层激活对应多分类问题

x = tf.linspace(-6.,6.,10)
# sigmoid激活函数  sigmoid(x) = e^x/(1+e^x)  输出层激活对应二分类问题
print(x)
print(tf.nn.sigmoid(x))

# relu激活函数 relu(x) =  max(0,x)
print(tf.nn.relu(x))


# leakey relu
print("leakey relu:",tf.nn.leaky_relu(x))
# print(tf.nn.leaky_relu(x,alpha=0.01))

# tanh激活函数 tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))
print("tanh:",tf.nn.tanh(x))
