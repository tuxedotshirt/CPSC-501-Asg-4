#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

print("--Get data--")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("--Process data--")
x_train, x_test = x_train / 255.0, x_test / 255.0
print("--Make model--")
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
#model.add(Dropout(rate=0.1))
model.add(Dense(10,activation='softplus'))
model.add(Dense(10,activation='softplus'))
#model.add(Dropout(rate=0.1))
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dropout(rate=0.1),
#  tf.keras.layers.Flatten(input_shape=(28, 28)),
#  tf.keras.layers.Dense(10, activation='softplus'),
#  tf.keras.layers.Dense(10, activation='softplus')
#])

adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 200
early_stop = EarlyStopping(monitor='accuracy',mode='max',verbose=1,patience=10)

print("--Fit model--")
model.fit(x_train, y_train, epochs=EPOCHS, verbose=1, callbacks=[early_stop])

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=0)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

#Save Model
model.save("MINST.h")
