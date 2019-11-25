# __author__:zsshi
# __date__:2019/11/25

import tensorflow as tf


#创建张量
# a = tf.convert_to_tensor([1,2.])
# print(a)
# b = tf.convert_to_tensor(np.array([1,2.]))#numpy默认使用64位精度
# print(b)

#创建全0，全1张量
# a = tf.zeros([])
# print(a)
# b = tf.ones([])
# print(b)
#
# a1 = tf.zeros([1])
# b1 = tf.ones([2])
# print(a1)
# print(b1)
#
# a2 = tf.zeros([2,3])
# b2 = tf.ones([1,2])
# print(a2,b2)

#创建自定义数值张量
# a = tf.fill([2,3],-1)
# print(a)

#创建已知分布的张量
# a = tf.random.normal([2,2])
# print(a)
# b = tf.random.normal([2,2],mean=1,stddev=2)
# print(b)

# a1 = tf.random.uniform([2,2],minval=0,maxval=3,dtype=tf.int32)
# b1 = tf.random.uniform([2,2],dtype=tf.float32)
# print(a1)
# print(b1)

#创建序列
a = tf.range(10)
print(a)
b = tf.range(10.)
print(b)
c = tf.range(10,delta=2)
print(c)