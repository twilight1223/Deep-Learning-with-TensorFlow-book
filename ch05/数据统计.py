# __author__:zsshi
# __date__:2019/11/26

import tensorflow as tf
import numpy as np
# 范数
# a = tf.random.normal([3])
# print(a)
# b1 = tf.norm(a,ord=1)
# print(b1)
# b2 = tf.norm(a,ord=2)
# print(b2)
# b3 = tf.norm(a,ord=np.inf)
# print(b3)

# 最大最小值，均值，和
# a = tf.random.normal([3,4])
# print(a)
# a_max = tf.reduce_max(a,axis=0)
# print(a_max)
# a_min = tf.reduce_min(a,axis=0)
# print(a_min)
# a_mean = tf.reduce_mean(a,axis=0)
# print(a_mean)
# a_sum = tf.reduce_sum(a,axis=0)
# print(a_sum)
#
# print(tf.reduce_max(a),tf.reduce_min(a),tf.reduce_mean(a),tf.reduce_sum(a))

# out = tf.random.normal([4,10])
# y = tf.constant([2,3,1,0])
# y = tf.one_hot(y,depth=10)
# loss = tf.keras.losses.mse(y,out)#计算每个样本的mse误差
# print(loss)
# loss = tf.reduce_mean(loss)
# print(loss)

# 求最值对应的索引
# out = tf.random.uniform([2,10],minval = 2,maxval = 10, dtype=tf.float32)
# print(out)
# # with tf.device("/cpu:0"):
# out = tf.nn.softmax(out,axis=1)#只能对float类型进行转换
# print(out)
# pred = tf.argmax(out,axis=1)#取最大值的索引作为类别预测
# print(pred)

# 张量比较
out = tf.random.uniform([100,10])
out = tf.nn.softmax(out,axis=1)
pred = tf.argmax(out,axis=1)
print(pred)
y = tf.random.uniform([100],dtype=tf.int64,maxval=10)
print(y)

equal_y = tf.equal(pred,y)
print("equal_y:",equal_y)
sum_count = tf.reduce_sum(tf.cast(equal_y,dtype=tf.float32))
print(sum_count)
acc = sum_count/100
print("acc:",acc)

# tf.math.greater
# tf.math.less
# tf.math.greater_equal
# tf.math.less_equal
# tf.math.not_equal
# tf.math.is_nan





