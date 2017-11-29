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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import class_weight
import os

np.random.seed(1337)


def split_data(X, y, test_data_size):
    '''
    Split data into test and training datasets.

    INPUT
        X: NumPy array of arrays
        y: Pandas series, which are the labels for input array X
        test_data_size: size of test/train split. Value from 0 to 1

    OUPUT
        Four arrays: X_train, X_test, y_train, and y_test
    '''
    return train_test_split(X, y, test_size=test_data_size, random_state=42)


def reshape_data(arr, img_rows, img_cols, channels):
    '''
    Reshapes the data into format for CNN.

    INPUT
        arr: Array of NumPy arrays.
        img_rows: Image height
        img_cols: Image width
        channels: Specify if the image is grayscale (1) or RGB (3)

    OUTPUT
        Reshaped array of NumPy arrays.
    '''
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def cnn_model(X_train, X_test, y_train, y_test, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes):
    '''
    Define and run the Convolutional Neural Network

    INPUT
        X_train: Array of NumPy arrays
        X_test: Array of NumPy arrays
        y_train: Array of labels
        y_test: Array of labels
        kernel_size: Initial size of kernel
        nb_filters: Initial number of filters
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification

    OUTPUT
        Fitted CNN model
    '''

    model = Sequential()


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
        padding='valid',
        strides=4,
        input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2,2)))


    kernel_size = (16,16)
    model.add(Conv2D(64, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))


    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)


    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


    model.compile(loss = 'categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


    stop = EarlyStopping(monitor='val_acc',
                            min_delta=0.001,
                            patience=2,
                            verbose=0,
                            mode='auto')


    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


    model.fit(X_train,y_train, batch_size=batch_size, epochs=nb_epoch,
                verbose=1,
                validation_split=0.2,
                class_weight=weights,
                callbacks=[stop, tensor_board])

    return model


def save_model(model, score, model_name):
    '''
    Saves Keras model to an h5 file, based on precision_score

    INPUT
        model: Keras model object to be saved
        score: Score to determine if model should be saved.
        model_name: name of model to be saved
    '''

    if score >= 0.75:
        print("Saving Model")
        model.save("../models/" + model_name + "_recall_" + str(round(score,4)) + ".h5")
    else:
        print("Model Not Saved.  Score: ", score)


if __name__ == '__main__':

    # Specify GPU's to Use
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

    # Specify parameters before model is run.
    batch_size = 1000
    nb_classes = 5
    nb_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3
    nb_filters = 32
    kernel_size = (8,8)

    # Import data
    labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv")
    X = np.load("../data/X_train_256_v2.npy")
    y = np.array(labels['level'])


    # Class Weights (for imbalanced classes)
    print("Computing Class Weights")
    weights = class_weight.compute_class_weight('balanced', np.unique(y), y)


    print("Splitting data into test/ train datasets")
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)


    print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)


    input_shape = (img_rows, img_cols, channels)


    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255


    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)


    print("Training Model")


    model = cnn_model(X_train, X_test, y_train, y_test, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes)


    print("Predicting")
    y_pred = model.predict(X_test)


    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    y_pred = [np.argmax(y) for y in y_pred]
    y_test = [np.argmax(y) for y in y_test]


    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')


    print("Precision: ", precision)
    print("Recall: ", recall)


    save_model(model=model, score=recall, model_name="DR_Two_Classes")


    print("Completed")
