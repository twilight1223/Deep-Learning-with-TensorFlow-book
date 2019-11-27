# __author__:zsshi
# __date__:2019/11/27

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
# data_path = tf.keras.utils.get_file('auto-mpg.data','http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/autompg.data')
column_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin']
df = pd.read_csv('auto-mpg.data',sep=' ',comment='\t',na_values='?',names=column_names,skipinitialspace=True)

dataset = df.copy()
# 预处理
print("缺失值统计：",dataset.isna().sum())
dataset = dataset.dropna()
print("缺失值统计：",dataset.isna().sum())

origin = dataset.pop('origin')
dataset['Usa'] = (origin==1)*1.0
dataset['Europe'] = (origin==2)*1.0
dataset['Japan'] = (origin==3)*1.0


# 统计分析,查看特征分布
plt.figure()
ax1 = plt.subplot(1,3,1)
ax1.scatter(dataset['cylinders'],dataset['mpg'])
ax2 = plt.subplot(1,3,2)
ax2.scatter(dataset['weight'],dataset['mpg'])
ax3 = plt.subplot(1,3,3)
ax3.plot(dataset['mpg'])
plt.show()




# 数据集准备
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('mpg')
test_labels = test_dataset.pop('mpg')

train_stas = train_dataset.describe()
train_stas = train_stas.transpose()


#训练和测试数据标准化都用train的统计信息
def norm(x):
    return (x-train_stas['mean'])/train_stas['std']
normed_train_stas = norm(train_dataset)
normed_test_stas = norm(test_dataset)

print(normed_train_stas.head())
print(normed_test_stas.head())

print(normed_train_stas.shape)
print(normed_test_stas.shape)

train_db = tf.data.Dataset.from_tensor_slices((normed_train_stas.values,train_labels.values))
train_db = train_db.shuffle(100).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((normed_test_stas.values,test_labels.values))
test_db = test_db.shuffle(100).batch(32)


# 搭建网络

class MyNetWork(tf.keras.Model):
    def __init__(self):
        super(MyNetWork,self).__init__()
        self.fc1 = tf.keras.layers.Dense(64,tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(64,tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(1)
    #重写call方法
    def call(self,inputs,training=None,mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 训练及测试
model = MyNetWork()
model.build(input_shape=(32,9))
print(model.summary())
train_losses = []
test_losses = []
optimizer = tf.keras.optimizers.RMSprop(0.001)
for epoch in range(200):
    # 训练阶段
    train_ls = []
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.keras.losses.MSE(out,y)
            loss = tf.reduce_mean(loss)
            # mae_loss = tf.reduce_mean(tf.keras.losses.MAE(out,y))
        # if step%5==0:
        print("current epoch:%d train step:%d loss:%f" %(epoch,step,loss))
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        train_ls.append(loss.numpy())
    train_ls_one_epoch = np.mean(train_ls)
    train_losses.append(train_ls_one_epoch)

    # 测试阶段
    # 一个 epoch 训练之后，计算测试集误差
    test_ls = []
    for (test_x,test_y) in test_db:
        test_out = model(test_x)
        test_loss = tf.keras.losses.MSE(test_out,test_y)
        test_loss = tf.reduce_mean(test_loss)#计算一个batch的平均误差
        test_ls.append(test_loss.numpy())
    test_ls_one_epoch = np.mean(test_ls)
    test_losses.append(test_ls_one_epoch)
plt.plot(train_losses,label='train')
plt.plot(test_losses,label='test')
plt.legend()
plt.savefig('mpg_loss.svg')
plt.show()







