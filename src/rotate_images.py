import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import rotate
import cv2
import os
import time

def rotate_images(file_path, degrees_of_rotation, lst_imgs):
    '''
    Rotates image based on a specified amount of degrees

    INPUT
        file_path: file path to the folder containing images.
        degrees_of_rotation: Integer, specifying degrees to rotate the
        image. Set number from 1 to 360.
        lst_imgs: list of image strings.

    OUTPUT
        Images rotated by the degrees of rotation specififed.
    '''

    for l in lst_imgs:
        img = io.imread(file_path + str(l) + '.jpeg')
        img = rotate(img, degrees_of_rotation)
        io.imsave(file_path + str(l) + '_' + str(degrees_of_rotation) + '.jpeg', img)


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
    start_time = time.time()
    trainLabels = pd.read_csv("../labels/trainLabels_master.csv")

    trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
    trainLabels_no_DR = trainLabels[trainLabels['level'] == 0]
    trainLabels_DR = trainLabels[trainLabels['level'] >= 1]

    lst_imgs_no_DR = [i for i in trainLabels_no_DR['image']]
    lst_imgs_DR = [i for i in trainLabels_DR['image']]

    # lst_sample = [i for i in os.listdir('../data/sample/') if i != '.DS_Store']
    # lst_sample = [str(l.strip('.jpeg')) for l in lst_sample]


    # Mirror Images with no DR one time
    print("Mirroring Non-DR Images")
    mirror_images('../data/train-resized-256/', 1, lst_imgs_no_DR)


    # Rotate all images that have any level of DR
    print("Rotating 90 Degrees")
    rotate_images('../data/train-resized-256/', 90, lst_imgs_DR)

    print("Rotating 120 Degrees")
    rotate_images('../data/train-resized-256/', 120, lst_imgs_DR)

    print("Rotating 180 Degrees")
    rotate_images('../data/train-resized-256/', 180, lst_imgs_DR)

    print("Rotating 270 Degrees")
    rotate_images('../data/train-resized-256/', 270, lst_imgs_DR)

    print("Mirroring DR Images")
    mirror_images('../data/train-resized-256/', 0, lst_imgs_DR)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))
