#%%
import tensorflow as tf
assert tf.__version__.startswith('2.')

# 2.0支持命令式编程

# 1.创建输入张量
a = tf.constant(2.)
b = tf.constant(4.)
# 2.直接计算并打印
print('a+b=',a+b)


