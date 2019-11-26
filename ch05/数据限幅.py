# __author__:zsshi
# __date__:2019/11/26

import tensorflow as tf

"""
类似relu的功能
"""

a = tf.random.normal([10],dtype = tf.float32)
print("原始数据：",a)
b = tf.maximum(a,0)
print("上限幅数据：",b)
c = tf.minimum(a,0)
print("下限幅数据：",c)

d = tf.minimum(tf.maximum(a,-0.1),0.1)
print("上限幅：0.1，下限幅：-0.1->",d)

e = tf.clip_by_value(a,-0.1,0.1)
print(e)