# __author__:zsshi
# __date__:2019/11/27

import tensorflow as tf

#均方差
out = tf.random.normal([2,10])
y = tf.random.uniform([2],maxval=10,dtype=tf.int32)
y = tf.one_hot(y,depth=10)
loss = tf.keras.losses.MSE(out,y)
print(loss) #两个样本的误差
print(tf.reduce_mean(loss)) #batch误差

# 交叉熵
"""
（1）信息熵，表示信息的不纯度，熵越小，不纯度越小，对于确定问题，熵为0
 (2) 交叉熵，信息熵与KL散度之和，表征两个分布之间的距离
 H(y,o) = H(y)+D_kl(y|o) = D_kl(y|o) = -log(o)  H(y)的分布确定,最小化交叉熵的过程也就是最大化正确类别概率预测的过程
"""
#


