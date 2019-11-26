#%%
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets
import  os


# 合并concat：不会产生新的维度
# a = tf.random.normal([4,35,8]) # 模拟成绩册A
# b = tf.random.normal([6,35,8]) # 模拟成绩册B
# c = tf.concat([a,b],axis=0) # 合并成绩册
# print(c.shape)
#
# a1 = tf.random.normal([4,35,4])
# b1 = tf.random.normal([4,35,4])
# c1 = tf.concat([a1,b1],axis=2)
# print(c1.shape)

# 堆叠stack：会产生新的维度，需要指定新维度的位置

# a = tf.random.normal([35,8])
# b = tf.random.normal([35,8])
# c = tf.stack([a,b],axis=0)
# c2 = tf.stack([a,b],axis=-1)
# print(c.shape)
# print(c2.shape)


# 分割split
a = tf.random.normal([10,32,8])
# # print(a.shape)
# # c = tf.split(a,axis=0,num_or_size_splits=10)
# # print(c)#tensor列表
# c2 = tf.split(a,axis=0,num_or_size_splits=[2,4,2,2])
# print(c2)

# unstack：在指定维度上按长度为1进行切割
c3 = tf.unstack(a,axis=0)
print(c3)

