#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('~/Asg4/heart.csv',true_values=['Present'],false_values=['Absent'])
df = df.drop('row.names',axis=1)

#check for number of rows and null entries
print('Print Info')
df.info()

#check for obvious correlation
print('Transpose')
df.describe().transpose()

#check for balanced data
print('Plot CHD')
sns.countplot(x='chd',data=df)

#plot correlation
print('Plot Correlation')
df.corr()['chd'][:-1].sort_values().plot(kind='bar')

#drop chd column to train fit
X= df.drop('chd',axis=1).values

y=df['chd'].values

#split train and test values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=19)

#scale values to normalize them
scaler = MinMaxScaler()

#fit values without chd so test is actually a test
X_train = scaler.fit_transform(X_train)

#fill test
X_test = scaler.transform(X_test)

#double check that the chd column is not included
X_train.shape

###############################################################
#explore overfit
model = Sequential()
model.add(Dense(9,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='relu'))
model.add(Dropout(0.5))
#binary classification = sigmoid
model.add(Dense(1,activation='sigmoid')) #somewhere between 0 and 1
adam = tf.keras.optimizers.Adam(0.001)
model.compile(loss='mse',optimizer=adam)

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

model.predict_classes(X_test)

predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
###################################################################3

#best model
model = Sequential()

model.add(Dense(9,activation='relu'))#relu
model.add(Dropout(0.2))
model.add(Dense(9,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(5,activation='relu'))

#binary classification = sigmoid
model.add(Dense(1,activation='sigmoid')) #somewhere between 0 and 1
rms = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse',optimizer=rms, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

model.predict_classes(X_test)

predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))

model.save("3.h")
