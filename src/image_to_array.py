import time

import numpy as np
import pandas as pd
from PIL import Image


def change_image_name(df, column):
    """
    Appends the suffix '.jpeg' for all image names in the DataFrame

    INPUT
        df: Pandas DataFrame, including columns to be altered.
        column: The column that will be changed. Takes a string input.

    OUTPUT
        Pandas DataFrame, with a single column changed to include the
        aforementioned suffix.
    """
    return [i + '.jpeg' for i in df[column]]


def convert_images_to_arrays_train(file_path, df):
    """
    Converts each image to an array, and appends each array to a new NumPy
    array, based on the image column equaling the image file name.

    INPUT
        file_path: Specified file path for resized test and train images.
        df: Pandas DataFrame being used to assist file imports.

    OUTPUT
        NumPy array of image arrays.
    """

    lst_imgs = [l for l in df['train_image_name']]

    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])


def save_to_array(arr_name, arr_object):
    """
    Saves data object as a NumPy file. Used for saving train and test arrays.

    INPUT
        arr_name: The name of the file you want to save.
            This input takes a directory string.
        arr_object: NumPy array of arrays. This object is saved as a NumPy file.

    OUTPUT
        NumPy array of image arrays
    """
    return np.save(arr_name, arr_object)


if __name__ == '__main__':
    start_time = time.time()

    labels = pd.read_csv("../labels/trainLabels_master_256_v2.csv")

    print("Writing Train Array")
    X_train = convert_images_to_arrays_train('../data/train-resized-256/', labels)

    print(X_train.shape)

    print("Saving Train Array")
    save_to_array('../data/X_train.npy', X_train)

    print("--- %s seconds ---" % (time.time() - start_time))
