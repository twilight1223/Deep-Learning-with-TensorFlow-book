# __author__:zsshi
# __date__:2019/11/26

import tensorflow as tf
# tf.gather:根据索引号收集数据

# a = tf.random.uniform([4,32,8],maxval=100,dtype=tf.int32)
# b = tf.gather(a,[0,1],axis=0)
# print(b)
#
# # 收集第 1,4,9,12,13,27 号同学成绩
# c = tf.gather(a,[0,3,8,11,12,26],axis=1)
# print(c)
#
# d = tf.gather(a,[2,4],axis=2)
# print(d)

# 乱序收集
# a = tf.range(8)
# a = tf.reshape(a,[4,2])
# print(a)
# b = tf.gather(a,[3,0,1,2])
# print(b)

#多维收集
# a = tf.random.uniform([4,32,8])
## 收集第[2,3]班级[3,4,6,27]号同学成绩
# students = tf.gather(a,[2,3,5,26],axis=1)
#
# class_23 = tf.gather(students,[1,2],axis=0)
# print(class_23)

# 采样第i班级，第j学生，第k科目成绩
# sample = tf.gather_nd(a,[[1,1,1],[3,5,7]])
# print(sample)

#掩码采样
# sample_mask = tf.boolean_mask(a,mask=[True,False,False,True],axis=0)
# print(sample_mask)

# 采样第1个班级的1-2号学生，第2个班级的2-3号学生
# a = tf.random.uniform([2,3,6],maxval=100,dtype=tf.int32)
# print(a)
# mask = [[True,True,False],[False,True,True]]
# scores = tf.boolean_mask(a,mask=mask)
# print(scores)

# where (cond,a,b),构造卷积核,水平特征提取
# a = tf.zeros([3,3])
# b = tf.ones([3,3])
# cond = [[False,False,False],[True,True,True],[False,False,False]]
# c = tf.where(cond,b,a)
# print(c)
# print(tf.where(cond))

# 提取出数据和索引，及用gather采样
# a = tf.random.normal([4,4])
# print(a)
# mask = a>0
# indices = tf.where(mask)
# print(indices)
# sample = tf.gather_nd(a,indices)
# print(sample)

# 在白板数据上刷新数据
# indices = tf.constant([[0],[3],[1],[9]])
# updates = tf.constant([1.2,3.6,4.3,9.0])
# a = tf.scatter_nd(indices,updates,[10])
# print(a)

# indices = tf.constant([[1],[3]])#indices与updates维度一致
# updates = tf.constant([[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]])
# a = tf.scatter_nd(indices,updates,[4,4,4])
# print(a)

# meshgrid
# 三维网格绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
ax = Axes3D(fig)

# data_points = []
#
# for x in np.linspace(-8,8,100):
#     for y in np.linspace(-8,8,100):
#         z = np.sqrt(x**2+y**2)
#         z = (np.sin(z))/z
#         data_points.append([x,y,z])
#

# x = np.arange(5)
# y = np.arange(5)
# z = np.arange(25).reshape(5, 5)
# x1, y1 = np.meshgrid(x, y)
# plt.contour(x1, y1, z)
# print(x1)
# print(y1)
# print(z)
# ax.contour3D(x1,y1,z)
# x1 = np.array([x[0] for x in data_points]).reshape(100,100)
# print(x1[0][0])
# print(x1)
# y1 = np.array([x[1] for x in data_points]).reshape(100,100)
# print(y1[0][0])
# print(y1)
# z1 = np.array([x[2] for x in data_points]).reshape(100,100)
# print(z1[0][0])
# print(z1)

x = tf.linspace(-8.,8,100)
y = tf.linspace(-8.,8,100)
x,y = tf.meshgrid(x,y)
z = x**2+y**2
z = tf.sin(z)/z


ax.contour3D(x.numpy(),y.numpy(),z.numpy())
plt.savefig('plt3d.svg')
plt.show()






