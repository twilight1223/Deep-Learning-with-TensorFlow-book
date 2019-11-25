# __author__:zsshi
# __date__:2019/11/25

import tensorflow as tf

x = tf.random.normal([4,32,32,3])
#取第一张图片数据
# a1 = x[0]
# #取第一张第二张数据
# a2 = x[0][1]
# #取第一张第二行第三列数据
# a3 = x[0][1][2]
# #取第一张第二行第三列第二通道数据
# a4 = x[0][1][2][1]
# print(a1)
# print(a2)
# print(a3)
# print(a4)
#
# print(x[0,1,2,1])

#索引
# b = x[1:3]
# print(b.shape)
# print(b)
# print(x[0,::])
# print(x[:,0:32:2,0:32:2,:])#隔行采样，隔列采样
print(x[0:2,...,1:])#读取前两张图片G/B通道所有元素