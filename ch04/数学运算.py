# __author__:zsshi
# __date__:2019/11/25

import tensorflow as tf

# a = tf.range(5)
# b = tf.constant(2)
# print(a//b)
# print(a%b)
#
# c = tf.pow(a,2)
# print("c:",c)
# print("c2:",tf.square(a))
#
# a = tf.cast(a,tf.float32)
# d = tf.pow(a,0.5)
# print("d:",d)
# e = tf.sqrt(a)
# print("e:",e)
#
# # exp
#
# x = tf.random.uniform([1,4])
# exp_x = tf.exp(x)
#
# print("exp:",tf.exp(x))
# print("x:",x)
# # log
# # x = 10**x
# print("log:",tf.math.log(exp_x))


#矩阵相乘
# a = tf.random.uniform([2,3,28,32])
# b = tf.random.uniform([2,3,32,2])
# print(a@b)

#矩阵相乘支持自动broadcasting
a = tf.random.normal([4,28,32])
b = tf.random.normal([32,2])
# print(a@b)
print(tf.matmul(a,b))