# __author__:zsshi
# __date__:2019/11/28

from tensorflow.keras import layers,Sequential
from tensorflow.keras import optimizers,losses
import tensorflow as tf

model = Sequential([
    layers.Dense(12,tf.nn.sigmoid),
    layers.Dense(10,tf.nn.softmax)
])
x = tf.random.normal([10,10])
y = tf.constant([1,2,3,4,5,6,7,8,9,0])
y = tf.one_hot(y,depth=10)
out = model(x)
mse = tf.keras.losses.MSE(out,y)
print(mse)
print(model.summary())

for v in model.trainable_variables:
    print(v.name,v.shape)

model.compile(optimizer = optimizers.Adam(),
              loss = losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x,y,epochs=5)
print(history.history)



