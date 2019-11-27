# __author__:zsshi
# __date__:2019/11/27

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

"""
梯度优化算法验证函数  f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2
"""
def himmelblau(x):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
X,Y = np.meshgrid(x,y)
Z = himmelblau([X,Y])
fig = plt.figure('himmelblau')
ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# 计算解析解
x = tf.constant([4.,0.])
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y,[x])[0]
    x-=0.01*grads
print(x)




