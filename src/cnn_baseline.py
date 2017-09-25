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
# from keras.utils import plot_model

from sklearn.model_selection import train_test_split


np.random.seed(1337)  # for reproducibility


labels = pd.read_csv("../labels/trainLabels_master_256.csv")
X = np.load("../data/X_train_256.npy")
# X_test = np.load("../data/X_test.npy")
y = np.array([1 if l >= 1 else 0 for l in labels['level']])
# y_train = np.array(labels['level'])

print("Splitting data into test/ train datasets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


batch_size = 1000
nb_classes = 2
nb_epoch = 10

img_rows, img_cols = 256, 256
channels = 3
nb_filters = 6
pool_size = (2, 2)
kernel_size = (6, 6)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)

input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)


# CNN Model
model = Sequential()
model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
    padding = 'valid',
    strides = 1,
    input_shape = (img_rows,img_cols,channels)))
# model.add(BatchNormalization())

model.add(Activation('relu'))

kernel_size = (2,2)

model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
# model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.15))

model.add(Flatten())
print('Model flattened out to ', model.output_shape)

model.add(Dense(128))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.15))

model.add(Dense(64))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.15))

model.add(Dense(64))
# model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.15))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# model.add(BatchNormalization())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.summary()
# plot_model(model, to_file='../images/cnn_baseline_model.png')

model.compile(loss = 'categorical_crossentropy',
    optimizer=sgd, #'adam'
    metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='acc', #val_loss
    min_delta=0.001, patience=0, verbose=0, mode='auto')

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=None, write_graph=True, write_images=True)

model.fit(X_train, y_train, batch_size = batch_size, epochs=nb_epoch, validation_split = 0.2,
    verbose=1,
    # validation_data=(X_test, y_test),
    class_weight = 'auto', shuffle=True, callbacks = [earlyStopping, tbCallBack])


score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1]) # this is the one we care about

# prediction = model.predict(X_test)
#

# if __name__ == '__main__':
#     pass
