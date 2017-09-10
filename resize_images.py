import os
import sys
from PIL import Image


def resize_images(path, new_path):
    for item in dirs:
        if os.path.isfile(path+item) and not item.startswith('.'):
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            img_resize = img.resize((120,120), Image.ANTIALIAS)
            new_path = 'sample-resized/'
            # img_resize.save(f + '_resized.jpeg', 'JPEG')
            img_resize.save(os.path.join(new_path,item))


if __name__ == '__main__':
    file_path = "sample/"
    new_path = "sample-resized/"
    dirs = os.listdir(file_path)
    resize_images(path = file_path, new_path = new_path)
