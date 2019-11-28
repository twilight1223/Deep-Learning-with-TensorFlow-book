# __author__:zsshi
# __date__:2019/11/28

from tensorflow.keras import layers,Sequential
from tensorflow.keras import optimizers,losses
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

X,y = datasets.make_moons(n_samples=2000,random_state=100)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

model = Sequential([
    layers.Dense(12,tf.nn.relu),
    layers.Dense(2,tf.nn.sigmoid)
])
model.build(input_shape=(None,2))

model.compile(optimizer = optimizers.Adam(),
              loss = losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=5,batch_size=128,verbose=1)
print(history.history)




