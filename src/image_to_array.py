import cv2
import numpy as np
import pandas as pd
import os
import pickle
import sys

def change_image_name(df, column):
    '''
    Appends the suffix '.jpeg' for all image names in the DataFrame

    INPUT
        df: Pandas DataFrame, including columns to be altered.
        column: The column that will be changed. Takes a string input.

    OUTPUT
        Pandas DataFrame, with a single column changed to include the
        aforementioned suffix.
    '''
    return [i + '.jpeg' for i in df[column]]


def convert_images_to_arrays(file_path, df):
    '''
    Converts each image to an array, and appends each array to a new NumPy
    array, based on the image column equaling the image file name.

    INPUT
        file_path: Specified file path for resized test and train images.
        df: Pandas DataFrame being used to assist file imports.

    OUTPUT
        NumPy array of image arrays.
    '''
    # lst_images = [i for i in os.listdir(file_path) if not i.startswith('.')]
    arr = np.empty(shape=(df.shape[0],120,120,3))

    # for i in labels_sample['image']:
    #     img = cv2.imread('../sample-resized/' + i)
    #     X_train.append(img)

    for i in df['image']:
        img = cv2.imread(file_path + i)
        np.append(arr, img)
    return arr

def save_to_pickle(py_object, pickle_name):
    '''
    Saves data object to Pickle file. Used for saving the train and test data.

    INPUT
        py_object: Data that will be saved to Pickle file.
        file_path: Name specified for writing pickle file.
            You can also specify the directory to save the file.

    OUTPUT
        Pickle file of train and test data.
    '''
    pickle.dump(py_object, open(pickle_name, 'wb'))


if __name__ == '__main__':

    labels = pd.read_csv("../labels/trainLabels.csv")

    labels.image = change_image_name(labels, 'image')

    # labels_sample = labels.head(10)

    # For each image, read in, save to Pandas DataFrame
    X_train = convert_images_to_arrays('../data/train-resized/', labels)
    X_test = convert_images_to_arrays('../data/test-resized/', labels)

    # X_train = convert_images_to_arrays('../sample-resized/', labels_sample)

    save_to_pickle(X_train, '../data/X_train.pkl')
    save_to_pickle(X_test, '../data/X_test.pkl')
