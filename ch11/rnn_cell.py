# __author__:zsshi
# __date__:2019/11/28

import tensorflow as tf
from tensorflow.keras import layers,Sequential

# cell = layers.SimpleRNNCell(5)
# cell.build(input_shape=(None,6))
# # x = tf.random.normal([10,4])
# # h0 = tf.constant([3])
# # out = cell(x)
# # print(out)
# print(cell.trainable_variables)

# cell = layers.SimpleRNNCell(64)
x = tf.random.normal([4,80,64])
# h0 = [tf.zeros([4,64])]
# xt = x[:,0,:]#第一个单词向量
# print(xt)
# h = h0
# for xt in tf.unstack(x,axis=1):#循环
#     out,h1 = cell(xt,h)
#     print('---------------out-------------')
#     print(out.shape)
#     print('---------------h1--------------')
#     print(h1[0].shape)
# out = out
# print(out)

# 多层循环网络

model = Sequential([
    layers.SimpleRNNCell(64),
    layers.SimpleRNNCell(64)
])

h0 = [tf.zeros([4,64])]
h1 = [tf.zeros([4,64])]
for xt in tf.unstack(x,axis=1):
    out,h = model(xt,h0,h1)
print(out.shape)








