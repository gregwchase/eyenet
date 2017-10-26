import mxnet as mx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import class_weight
import os
import time

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


if __name__ == '__main__':
    # Specify GPU's to Use
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

    start = time.time()

    # Specify parameters before model is run.
    batch_size = 1000
    nb_classes = 2
    nb_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3

    # labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv", nrows = 60)
    labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv")
    X = np.load("../data/X_sample.npy")
    y = np.array(labels['level'])


    # Class Weights (for imbalanced classes)
    print("Computing Class Weights")
    weights = class_weight.compute_class_weight('balanced', np.unique(y), y)


    print("Splitting data into test/ train datasets")
    X_train, X_test, y_train, y_test = split_data(X, y, test_data_size=0.2)


    # print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)
    print(X_train.shape)
    print(X_test.shape)

    print("Importing To MXNet")
    batch_size = 5
    train = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
    test = mx.io.NDArrayIter(X_test, y_test, batch_size)


    print("Creating Model")

    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(1,1), num_filter=32)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(1,1))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=64)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(1,1))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=128)
    relu3 = mx.sym.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=5)
    # softmax loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')


    # create a trainable module on GPU 0
    # lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
    lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
    # train with the same
    print("Training")
    lenet_model.fit(train,
                    eval_data=test,
                    optimizer='sgd',
                    optimizer_params={'learning_rate':0.1},
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 10),
                    num_epoch=15)


    test_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)
    prob = lenet_model.predict(test_iter)
    # predict accuracy for lenet
    acc = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    # precision = mx.metric.Precision()
    lenet_model.score(test_iter, acc)
    # lenet_model.score(test_iter, precision)
    lenet_model.score(test_iter, f1)
    print(acc)
    print(f1)
    # assert acc.get()[1] > 0.98

    print("Seconds: ", round(time.time() - start, 2))
