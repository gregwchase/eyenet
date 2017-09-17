import os
import sys
from PIL import Image
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def resize_images_cv(path, new_path):
    for item in dirs:
        if os.path.isfile(path+item) and not item.startswith('.'):
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            img_resize = cv2.resize(img, (120, 120), interpolation = cv2.INTER_AREA)
            # img_resize = img.resize((120,120), Image.ANTIALIAS)
            created_path = new_path
            img_resize.save(os.path.join(created_path,item))


if __name__ == '__main__':
    resize_images(path = "../sample/", new_path='../sample-resized/', img_size=120)
    # resize_images(path = '../data/train', new_path='../data/train-resized')
    # resize_images(path = '../data/test', new_path='../data/test-resized')
