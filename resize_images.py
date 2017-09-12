import os
import sys
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resize_images(path, new_path):
    for item in dirs:
        if os.path.isfile(path+item) and not item.startswith('.'):
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            img_resize = img.resize((120,120), Image.ANTIALIAS)
            created_path = new_path
            img_resize.save(os.path.join(created_path,item))


if __name__ == '__main__':
    # Specify file paths
    file_path = "train_100/"
    new_path = "train_100_resized/"

    # Create the directory if it doesn't exist
    create_directory(new_path)

    # Creates a list of file strings
    dirs = os.listdir(file_path)

    # Resize the images, and save to the new directory
    resize_images(path = file_path, new_path = new_path)
