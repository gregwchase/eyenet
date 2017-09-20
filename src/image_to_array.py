import os
import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def image_to_array(path, img):
    '''
    Converts each image to an array, and appends array to the labels
    DataFrame, based on the image column equaling the image file name.

    INPUT


    OUTPUT


    '''
    lst_images = [i for i in os.listdir(path) if not i.startswith('.')]

if __name__ == '__main__':

    labels = pd.read_csv("../labels/trainLabels.csv")

    labels.image = change_image_name(labels, 'image')
    labels['img_array'] = None

    # Image to array code (insert into function when complete)
    lst_images = [i for i in os.listdir("../sample-resized") if not  i.startswith('.')]

    img_name = "15_left.jpeg"



    if img_name == [i for i in labels['image']]:
        labels['img_array'] = img

    labels['img_array'] = img.where(labels['image'] == img_name,0)

    # df['img_array'] = np.where(labels['image'] == img_name, img, 0)

    df.query('(a < b) & (b < c)')
