import os
import sys
from PIL import Image


def resize_images(path):
    for item in dirs:
        if os.path.isfile(path+item) and not item.startswith('.'):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((120,120), Image.ANTIALIAS)
            imResize.save(f + '_resized.jpg', 'JPEG')


if __name__ == '__main__':
    path = "sample/"
    new_path = "sample-resized/"
    dirs = os.listdir(path)
    resize_images(path = "sample/")
