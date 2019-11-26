# __author__:zsshi
# __date__:2019/11/26

import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import matplotlib

(x,y),(x_test,y_test) = datasets.mnist.load_data()
print("x",x.shape)
print("y",y.shape)
print("x_test",x_test.shape)
print("y_test",y_test.shape)

#转换成Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x,y))
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
print("train_db对象：",train_db)
print("test_db对象：",test_db)
#随机打散
train_db = train_db.shuffle(10000)
print("shuffle:",train_db)
#批训练
train_db = train_db.batch(128)
test_db = test_db.batch(128)
print("批处理：",train_db)

# 预处理
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    x = tf.reshape(x,[-1,28*28])
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y
train_db = train_db.map(preprocess)
test_db = test_db.map(preprocess)
print("train预处理map:",train_db)
print("test预处理map:",test_db)

# 初始化训练参数
w1 = tf.Variable(tf.random.normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))



# 循环训练
lr = 0.01
losses = []
accuracy = []
steps = []
train_db = train_db.repeat(2)
total = y_test.shape[0]
for step,(x,y) in enumerate(train_db):
    # 设置两层网络
    with tf.GradientTape() as tape:
        tape.watch([w1,b1,w2,b2,w3,b3])
        h1 = x@w1+b1
        h1 = tf.nn.relu(h1)
        h2 = h1@w2+b2
        h2 = tf.nn.relu(h2)
        h3 = h2@w3+b3
        out = tf.nn.relu(h3)
        loss = tf.reduce_mean(tf.square((y-out)))
    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    w1.assign_sub(lr*grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])
    losses.append(loss)
    steps.append(step)
    # 测试集评估
    total_correct = 0
    for x1,y1 in test_db:
        h1 = x1 @ w1 + b1
        h1 = tf.nn.relu(h1)
        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)
        h3 = h2 @ w3 + b3
        out = tf.nn.relu(h3)
        pred = tf.argmax(out,axis=1)
        y1 = tf.argmax(y1,axis=1)
        correct = tf.equal(pred,y1)
        total_correct += tf.reduce_sum(tf.cast(correct,dtype=tf.int32))/total
    accuracy.append(total_correct)

plt.plot(steps,losses)
plt.savefig("train_loss.svg")
plt.figure()
plt.plot(steps,accuracy)

plt.show()







# train_db = train_db.repeat(10)
# for step,(x,y) in enumerate(train_db):
#     print(step)
# print(train_db)




