# __author__:zsshi
# __date__:2019/11/25
import tensorflow as tf


#标量应用:误差值，各种张量指标的表示
# out = tf.random.uniform([4,10])
# print(out)
# y = tf.constant([1,2,3,4])
# y = tf.one_hot(y,depth=10)
# loss = tf.keras.losses.mse(y,out)
# result = tf.reduce_mean(loss)
# print(result)

#向量应用：偏置向量b
# z = tf.random.uniform([2,2])
# b = tf.zeros([2])
# z = z+b
# print(z)
# fc = tf.keras.layers.Dense(3)
# fc.build(input_shape=(2,4))
# print(fc.bias)

#矩阵，权重向量
#假定输入为4节点，输出为三节点，样本数为2
# x = tf.random.uniform([2,4])
# w = tf.ones([4,3])
# b = tf.zeros([3])
# o = x@w + b
# print(o)

# fc = tf.keras.layers.Dense(3)
# fc.build(input_shape=(2,4))
# print(fc.kernel)


#三维张量：序列信号shape 为[2,5,3]的 3 维张量，其中 2 表示句子个数， 5 表示单词数量， 3 表示单词向量的长度
# (x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
# x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
# print(x_train.shape)
# #(25000, 80)25000个句子，每个句子有80个词
# embedding = tf.keras.layers.Embedding(10000,100)
# out = embedding(x_train)
# print(out.shape)
#(25000, 80, 100)每个词被编码为长度为100的向量

#4维张量
x = tf.random.normal([4,32,32,3])
layer = tf.keras.layers.Conv2D(16,kernel_size=3)
out = layer(x)
print(out.shape)
print(out)




