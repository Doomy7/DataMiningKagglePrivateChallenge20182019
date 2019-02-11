# George Cheirmpos 3130230
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import manipulator as mp
import time
# from keras.models import load_model
# model = load_model('3130230_model_0.66017.h5', custom_objects={'f1': mp.f1})

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

kaggle = 1
if(kaggle):
    df_train, df_test, y_train, y_test = mp.kaggletest()
else:
    df_train, df_test, y_train, y_test = mp.localtest()

X_train, X_test = mp.masterManipulator(df_train, df_test, kaggle)

y_train = np.ravel(y_train)
unique, counts = np.unique(y_train, return_counts=True)
counts = counts/min(counts)

y_train = keras.utils.to_categorical(y_train, num_classes=8)

model = Sequential()
rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
tensorboard = TensorBoard(log_dir='./logs/{}'.format("Test{}".format(time.time())))

'''
Gave 2nd place private score 0.61691 with public score 0.66017, still overfiting
'''
model.add(Dense(224, activation='relu', input_dim=258))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(8, actication='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=rmsp, metrics=[mp.f1, 'accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=2, class_weight=counts, callbacks=[tensorboard])

model.save('3130230_model_' + str(time.time()) + '.h5')
y_pred = model.predict_classes(X_test)
mp.finalize(y_test, y_pred, kaggle)


'''
Gave highest personal public score (0,66467)-> (high overfit) private score 0.61050
model.add(Dense(224, activation='relu', input_dim=258))
model.add(Dropout(0.7))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(8, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=rmsp, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=256, class_weight=counts, verbose=2)
'''