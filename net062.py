import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Nadam
import manipulation as mp
from keras.callbacks import TensorBoard, EarlyStopping
import manipulator as mt
import sys
from tensorflow.contrib.metrics import f1_score
import tensorflow as tf
from collections import Counter
import random as rn
from keras import regularizers
import os
import time

np.random.seed(1)
tf.set_random_seed(2)

pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 175)
pd.set_option('display.max_rows', 30)
# tensorboard = TensorBoard(log_dir='logs/{}'.format("test"))
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


kaggle = 1
if(kaggle):
    df_train, df_test, y_train, y_test = mp.kaggletest()
else:
    df_train, df_test, y_train, y_test = mp.localtest()
droplist = [9, 10, 11]
column_dict = {'0': 'DateOfDeparture',
                '1': 'Departure', '2': 'CityDeparture',
                '3': 'LongitudeDeparture', '4': 'LatitudeDeparture',
                '5': 'Arrival', '6': 'CityArrival',
                '7': 'LongitudeArrival', '8': 'LatitudeArrival',
                '9': 'WeeksToDeparture', '10': 'std_wtd',
                 '11': 'PAX'}

X_train, X_test = mt.masterManipulator(droplist, column_dict, df_train, df_test, 0, kaggle)


y_train = np.ravel(y_train)
unique, counts = np.unique(y_train, return_counts=True)
counts = counts/min(counts)
print(counts)
print(unique)

y_train = keras.utils.to_categorical(y_train, num_classes=8)
model = Sequential()

rmsp = RMSprop(lr=0.00085, rho=0.9, epsilon=None, decay=0.0)
sgd = SGD(lr=.01, decay=1e-6, momentum=0.9, nesterov=True)
adad = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adag = Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#, kernel_regularizer=regularizers.l2(0.01)
tensorboard = TensorBoard(log_dir='./logs/{}'.format("Test{}".format(time.time())))
model.add(Dense(320, activation='relu', input_dim=258))
model.add(Dropout(0.75))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(8, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=rmsp, metrics=[mp.f1, 'accuracy'])
model.fit(X_train, y_train, epochs=110, batch_size=256, verbose=2, class_weight=counts, callbacks=[tensorboard])
#   callbacks=[monitor1

model.save('3130230_model.h5')
y_pred = model.predict_classes(X_test)
mp.finalize(y_test, y_pred, kaggle, "LogisticRegression")
