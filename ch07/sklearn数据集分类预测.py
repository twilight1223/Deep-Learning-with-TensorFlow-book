# __author__:zsshi
# __date__:2019/11/28

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

"""
目前没有跑通，还要再研究一下反向传播的变换过程
"""

#生成分类数据集，图形化展示
X,y = datasets.make_moons(n_samples=2000,noise=0.2,random_state=100)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


def make_plot(X,y,plot_name=None,file_name=None,XX=None,YY=None,preds=None,dark=False):
    #设置图形格式
    if dark:
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')

    plt.figure(figsize=(16,12))
    axes = plt.gca()#get axes instance
    axes.set(xlabel='x_1',ylabel='x_2')
    plt.title(plot_name,fontsize=30)
    plt.subplots_adjust(left=0.2,right=0.8)
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX,YY,preds.reshape(XX.shape),25,alpha=1,cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap='Greys',vmin=0,vmax=.6)

    plt.scatter(X[:,0],X[:,1],c=y.ravel(),s=40,cmap=plt.cm.Spectral,edgecolors='None')
    plt.savefig('dataset.svg')
    plt.close()
make_plot(X,y,plot_name='Classification Dataset Visualization')
plt.show()

#层模型
class Layer:
    def __init__(self,n_input,n_neurons,activation=None,weights=None,bias=None):
        self.weights = weights if weights is not None else tf.random.normal([n_input,n_neurons])/np.sqrt(1/n_neurons)
        self.bias = bias if bias is not None else tf.zeros([n_neurons])
        self.activation = activation
        self.last_activation = None #输出值
        self.error = None
        self.delta = None
    def activate(self,x):
        r = np.dot(x,self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self,r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r,0)

        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-r))

        elif self.activation == 'tanh':
            return np.tanh(r)
        else:
            return r
    #执行反向传播,计算不同激活函数的导数
    def apply_activation_devirate(self,r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r,copy=True)
            grad[r>0]=1
            grad[r<=0]=0
            return grad
        elif self.activation == 'tanh':
            return 1-r**2
        elif self.activation == 'sigmoid':
            return r*(1-r)
        else:
            return r

# 网络模型
class netWork:
    def __init__(self):
        self._layers=[]
    def add_layer(self,layer):
        self._layers.append(layer)

    def feed_forward(self,X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def backpropagation(self,X,y,learning_rate):
        out = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]: #输出层
                layer.error = y-out
                layer.delta = layer.error*layer.apply_activation_devirate(out)
            else:
                next_layer = self._layers[i+1]
                layer.error = np.dot(next_layer.weights,next_layer.delta)
                layer.delta = layer.error*layer.apply_activation_devirate(layer.last_activation)
        for i in range(len(self._layers)):
            o_i = np.atleast_2d(X if i == 0 else self._layers[i-1].last_activation)
            # 梯度下降算法， delta 是公式中的负数，故这里用加号
            layer.weights += layer.delta * o_i.T * learning_rate

    def accuracy(self,out,y):
        return np.mean(out==y)

    def predict(self,X):
        return self.feed_forward(X)

    def train(self,X_train, X_test, y_train, y_test, learning_rate,max_epochs=None):
        # one-hot编码
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses=[]
        for epoch in range(10):
            for i in range(X_train.shape[0]):
                self.backpropagation(X_train[i],y_onehot[i],learning_rate)
            if epoch % 10 == 0:
                mse = np.mean(np.square(y_onehot-self.feed_forward(X_train)))
                mses.append(mse)

                print("Epoch:%d MSE:%f" %(epoch),mse)
                print("Accuracy:%.2f%" % self.accuracy(self.predict(X_test),y_test.flatten()) * 100)





# 实例化网络对象
nn = netWork()
nn.add_layer(Layer(2,25,'sigmoid'))
nn.add_layer(Layer(25,50,'sigmoid'))
nn.add_layer(Layer(50,25,'sigmoid'))
nn.add_layer(Layer(25,2,'sigmoid'))

#训练评估

nn.train(X_train,X_test,y_train,y_test,learning_rate=0.01)
