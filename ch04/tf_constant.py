# __author__:zsshi
# __date__:2019/11/25

#constant包含tensor的基本属性，shape,dtype,numpy

import tensorflow as tf
# a = 1.2
# aa = tf.constant(1.2)
# print(type(a),type(aa))
# print(tf.is_tensor(aa))

# x = tf.constant([1.2,3.4])
# print(x)
# x = x.numpy()
# print(x,type(x))

#向量需通过列表传入
# a = tf.constant([[1.,2.],[3.,4.]])
# print(a)
# print(a.numpy())

#字符串类型
# a = tf.constant("Hello,Deep learning")
# print(a)
#
# b = tf.strings.lower(a)
# print(a)
# print(b)

# 布尔类型
# a = tf.constant(True)
# print(a)
# b = tf.constant([True,False])
# print(b)
# c = (a==True)
# print(c)

#数值精度
# a = tf.constant(123456789,dtype=tf.int16)
# b = tf.constant(123456789,dtype=tf.int32)
# print(a,b)
#
import numpy as np
pi = np.pi
# c = tf.constant(pi,dtype=tf.float32)
# print(c)
# d = tf.constant(pi,dtype=tf.float64)
# print(d)

#转换精度
# a_dtype = a.dtype
# print("before:",a_dtype)
# if a_dtype != tf.int64:
#     a = tf.cast(a,tf.int64)
# print("after:",a.dtype)
# print(a)

#类型转换
a = tf.constant(pi,dtype=tf.float16)
a = tf.cast(a,tf.double)
print(a)

b = tf.constant(123456789,dtype=tf.int32)
b = tf.cast(b,tf.int16)
print(b)

c = tf.constant([True,False])
c = tf.cast(c,tf.int32)
print(c)

d = tf.constant([1,2,3,4])
d = tf.cast(d,tf.bool)
print(d)


