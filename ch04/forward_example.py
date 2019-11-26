# __author__:zsshi
# __date__:2019/11/25
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib



"""采用的数据集是 MNIST 手写数字图片集，输入节点数为 784，第一层的输出节点数是
256，第二层的输出节点数是 128，第三层的输出节点是 10，也就是当前样本属于 10 类

out = relu{relu{relu{x@w1+b1}@w2+b2}@w3+b3}
"""
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

(x,y),_ = tf.keras.datasets.mnist.load_data()

x = 2*tf.convert_to_tensor(x, dtype=tf.float32) / 255.-1
x = tf.reshape(x,[-1,28*28])
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y,depth=10)

w1 = tf.Variable(tf.random.normal([784,256],mean=0,stddev=0.1))#w初始方差小的话训练速度很慢
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.normal([256,128],mean=0,stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.normal([128,10],mean=0,stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = tf.constant(0.01)
losses = []
iter =[]



for i in range(20):
    with tf.GradientTape() as tape:
        tape.watch([w1,b1,w2,b2,w3,b3])#把变量加入梯度跟踪列表
        h1 = x@w1+b1
        h1 = tf.nn.relu(h1)
        h2 = h1@w2+b2
        h2 = tf.nn.relu(h2)
        h3 = h2@w3+b3
        out = tf.nn.relu(h3)
        loss = tf.reduce_mean(tf.square(out-y))
        print(loss)

    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])
    # w1 = w1 - lr * grads[0]
    # b1 = b1 - lr * grads[1]
    # w2 = w2 - lr * grads[2]
    # b2 = b2 - lr * grads[3]
    # w3 = w3 - lr * grads[4]
    # b3 = b3 - lr * grads[5]

    losses.append(loss)
    iter.append(i)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(iter,losses,marker='s',label='训练')
plt.legend()
plt.savefig('forward.svg')
plt.show()


