import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import rotate
import cv2
import os

def rotate_image(file_path, degrees_of_rotation):
    '''
    Rotates image based on a specified amount of degrees

    INPUT
        file_path: file path to the folder containing images.
        degrees_of_rotation: Integer, specifying degrees to rotate the
        image. Set number from 1 to 360.

    OUTPUT
        Images rotated by the degrees of rotation specififed.
    '''

    img = io.imread('../data/sample/' + 'image_path_here')
    img = rotate(img, 90)
    io.imsave('img_name + _r_ + 90/180 + .jpeg', img)
    pass

def mirror_images(file_path, mirror_direction, lst_imgs):
    '''
    Mirrors image left or right, based on criteria specified.

    INPUT
        file_path: file path to the folder containing images.
        mirror_direction: criteria for mirroring left or right.
        lst_imgs: list of image strings.

    OUTPUT
        Images mirrored left or right.
    '''

    for l in lst_imgs:
        img = cv2.imread(file_path + str(l) + '.jpeg')
        img = cv2.flip(img, 1)
        cv2.imwrite(file_path + str(l) + '_mir' + '.jpeg', img)


if __name__ == '__main__':
    trainLabels = pd.read_csv("../labels/trainLabels_master.csv")

    trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
    trainLabels_no_DR = trainLabels[trainLabels['level'] == 0]
    trainLabels_DR = trainLabels[trainLabels['level'] >= 1]

    lst_imgs_no_DR = [i for i in trainLabels_no_DR['image']]
    lst_imgs_DR = [i for i in trainLabels_DR['image']]

    lst_sample = [i for i in os.listdir('../data/sample/') if i != '.DS_Store']
    lst_sample = [str(l.strip('.jpeg')) for l in lst_sample]

    mirror_images('../data/sample-resized/', 1, lst_sample)
