# __author__:zsshi
# __date__:2019/11/25

"""
自己实现一个回归模型
模型：y = 1.477*x + 0.089
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["lines.color"] = 'orange'
# 生成数据点[[x,y]]
data = []
for i in range(100):
    x = np.random.uniform(-10.,10.)
    eps = np.random.normal(0,0.1)
    y = 1.477 * x + 0.089 + eps
    data.append([x,y])
data = np.array(data)
# print(data)

def mse(w,b,points):
    sum_loss = []
    for point in points:
        loss = np.square(w * point[0] + b - point[1])
        sum_loss.append(loss)
    return np.mean(sum_loss)

def step_gradient(w_current,b_current,points,lr):
    """
    单步梯度更新
    w = w_current -lr * grad(w)
    b = b_current - lr * grad(b)

    grad(w) = 2/M *sum((w * x + b - y) * x)
    grad(b) = 2/M *sum((w * x + b - y))
    :param w_current:
    :param b_current:
    :param points:
    :param lr:
    :return:
    """
    w_gradient = 0
    b_gradient = 0
    for point in points:
        w_gradient += 2/len(points)*(w_current * point[0] + b_current - point[1]) * point[0]
        b_gradient += 2/len(points)*(w_current * point[0] + b_current - point[1])


    w_new = w_current - lr * w_gradient
    b_new = b_current - lr * b_gradient
    return [w_new,b_new]

def gradient_descent(starting_w,starting_b,points,lr,epoch_num):
    w_current = starting_w
    b_current = starting_b
    iterations = []
    losses = []

    for epoch in range(epoch_num):
        w_current,b_current = step_gradient(w_current,b_current,points,lr)
        loss = mse(w_current,b_current,points)
        if epoch % 50 == 0:
            print("current iteration:%s,loss:%s,w:%s,b:%s" %(epoch,loss,w_current,b_current))

        iterations.append(epoch)
        losses.append(loss)



    plt.plot(iterations,losses)
    plt.show()
    return [w_current,b_current]

def main():
    starting_w = 0
    starting_b = 0
    lr = 0.01
    epoch_num = 1000
    w,b = gradient_descent(starting_w,starting_b,data,lr,epoch_num)
    loss = mse(w,b,data)
    print("final_loss:%s,w:%s,b:%s" %(loss,w,b))

if __name__=="__main__":
    main()






