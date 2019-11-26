# __author__:zsshi
# __date__:2019/11/25

import tensorflow as tf
# reshape改变视图
#视图与存储，存储是按顺序逐个存储张量数据
# x = tf.random.normal([2,32,32,3])
# print(x.ndim,x.shape)
# a = tf.reshape(x,[2,-1])#-1指按维度推导法则自动推导维度
# print(a)

# expand_dims插入新维度
# x = tf.random.uniform([28,28])
# a = tf.expand_dims(x,axis=0)#axis指要扩展的维度
# print(a)
#
# # squeeze删除维度,指删除长度为1的维度
# b = tf.squeeze(a)
# print(b)

# x1 = tf.random.uniform([1,4,4,1],minval=0,maxval=20)
# print(x1)
# a1 = tf.squeeze(x1)
# print(a1)


# transpose交换维度
# 改变存储顺序
# 如改变通道数的维度顺序[b,h,w,c]->[b,c,h,w]
# x = tf.random.uniform([3,28,28,3])
# a = tf.transpose(x,perm=[0,3,1,2])#perm新维度的顺序
# print(a)

# tile复制数据
# b = tf.random.uniform([3])
# print(b)
# b = tf.expand_dims(b,axis=0)
# b = tf.tile(b,[2,1])
# print(b)

# broadcasting 优化，不需要IO复制
a = tf.random.uniform([1,4])
print(a)
b = tf.random.uniform([4,1])
print(b)
print(a+b)