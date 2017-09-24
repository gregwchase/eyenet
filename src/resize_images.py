import os
import sys
# from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np
import time


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.

    INPUT
        directory: Folder to be created, called as "folder/".

    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def resize_images(path, new_path, img_size):
    '''
    Resizes images based on the specified dimensions.
    Images are saves to the new_path directory.

    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.

    OUTPUT
        All images resized and saved from the old folder to the new folder.
    '''
    create_directory(new_path)
    dirs = os.listdir(path)

    for item in dirs:
        if os.path.isfile(path+item) and not item.startswith('.'):
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            img_resize = img.resize((img_size,img_size), Image.ANTIALIAS)
            created_path = new_path
            img_resize.save(os.path.join(created_path,item))

# def resize_images_cv(path, new_path):
#     for item in dirs:
#         if os.path.isfile(path+item) and not item.startswith('.'):
#             img = Image.open(path+item)
#             f, e = os.path.splitext(path+item)
#             img_resize = cv2.resize(img, (120, 120), interpolation = cv2.INTER_AREA)
#             # img_resize = img.resize((120,120), Image.ANTIALIAS)
#             created_path = new_path
#             img_resize.save(os.path.join(created_path,item))


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.

    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.

    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    total = 0

    for item in dirs:
        # img = io.imread('../data/sample/' + '13_left.jpeg')
        img = io.imread(path+item)
        # f,e = os.path.splitext(path+item)
        y,x,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[starty:starty+cropy,startx:startx+cropx]
        img = resize(img, (256,256))
        io.imsave(str(new_path + item), img)
        total += 1
        print("Saving: ", item, total)


if __name__ == '__main__':
    start_time = time.time()
    # resize_images(path = "../sample/", new_path='../sample-resized/', img_size=120)
    # resize_images(path = '../data/train/', new_path='../data/train-resized/')
    # resize_images(path = '../data/test/', new_path='../data/test-resized/')

    # crop_and_resize_images(path='../data/sample/', new_path='../data/sample-resized-256-1800/', cropx=1800, cropy=1800, img_size=256)
    crop_and_resize_images(path='../data/train/', new_path='../data/train-resized-256/', cropx=1800, cropy=1800, img_size=256)
    crop_and_resize_images(path='../data/test/', new_path='../data/test-resized-256/', cropx=1800, cropy=1800, img_size=256)
    # crop_and_resize_images(path='../train_100/', new_path='../data/train_100-resized-256-1800/', cropx=1800, cropy=1800, img_size=256)
    print(time.time() - start_time)
