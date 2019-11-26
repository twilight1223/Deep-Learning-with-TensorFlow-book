# __author__:zsshi
# __date__:2019/11/26

import tensorflow as tf
from scipy.misc import imread

# 使张量维度长度一致
#pad
# a = tf.random.normal([5])
# print(a)
# b = tf.random.normal([3])
# print(b)
# c = tf.pad(b,paddings=[[2,0]])
# print(c)
# #将填充后的向量堆叠在一起
# d = tf.stack([a,c],axis=0)
# print(d)

# 自然语言处理，句子的填充与截断处理
# TOTAL_WORDS = 10000
# SEQ_LEN = 80
# VECTOR_LEN = 100
# (x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=TOTAL_WORDS)
#
# # x_train = tf.convert_to_tensor(x_train) #目前还不能转为tensor，因为文本长度不一致
# x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=SEQ_LEN,truncating='post',padding='post')
# x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=SEQ_LEN,truncating='post',padding='post')
# print(x_train[0])
# print(x_test[0])
# x_train = tf.convert_to_tensor(x_train)
# x_test = tf.convert_to_tensor(x_test)
#
# print(x_train.shape,x_test.shape)


# 图片填充
# img = imread('2.png')
# print(img)
# img = tf.convert_to_tensor(img)
# print(img.shape)
# img = tf.pad(img,paddings=[[2,2],[2,2],[0,0]])
# print(img)

# 复制
a = tf.random.normal([4,32,32,3])
b = tf.tile(a,[2,3,3,1])
print(b.shape)





