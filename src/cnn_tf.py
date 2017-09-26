from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

np.random.seed(1337)

labels = pd.read_csv("../labels/trainLabels_master_256.csv")
X = np.load("../data/X_train_256.npy")
y = np.array([1 if l >= 1 else 0 for l in labels['level']])


print("Splitting data into test/ train datasets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


batch_size = 500
nb_classes = 2
nb_epoch = 20

img_rows, img_cols = 256, 256
channels = 3
nb_filters = 16
pool_size = (2,2)
kernel_size = (16,16)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)

input_shape = (img_rows, img_cols, channels)


y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)


'''
Create the CNN Model
'''
model = Sequential()


model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
    padding='valid',
    strides=4,
    input_shape=(img_rows, img_cols, channels)))
model.add(Activation('relu'))


kernel_size = (8,8)
model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))


kernel_size = (2,2)
model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.50))


model.add(Flatten())
print("Model flattened out to: ", model.output_shape)


model.add(Dense(128))
model.add(Activation('tanh'))


# model.add(Dense(32))
# model.add(Activation('tanh'))


# model.add(Dense(16))
# model.add(Activation('tanh'))


model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.summary()


model.compile(loss = 'categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


stop = EarlyStopping(monitor='acc',
                        min_delta=0.001,
                        patience=2,
                        verbose=0,
                        mode='auto')

model.fit(X_train,y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=1,
            validation_data=(X_test,y_test),
            # class_weight='auto',
            callbacks=[stop])


score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
