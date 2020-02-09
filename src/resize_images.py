
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
import psutil
from PIL import Image
from tqdm import tqdm

from correct_class_imbalance import split_data


class PreprocessImages:

    def __init__(self):
        self.df_labels = pd.read_csv("../data/trainLabels.csv")

    # TODO: Research NVIDIA DALI for preprocessing images
    # TODO: Speed of PIL vs OpenCV
    def make_image_thumbnail(self, filename):
        """
        Resize image for CNN

        INPUT
            filename: str, required; string of image to be processed
        
        OUTPUT
            Resized image, saved to directory
            String of filename with directory
        """
        # The thumbnail will be named "<original_filename>.jpeg"
        base_filename, file_extension = os.path.splitext(filename)
        thumbnail_filename = f"{base_filename}{file_extension}"
        
        thumbnail_filename = thumbnail_filename.replace("train", "train_resized")

        # Create and save thumbnail image
        # image = cv2.imread(filename)
        # image = cv2.resize(image, (256,256))
        # cv2.imwrite(f"{thumbnail_filename}", image)
        image = Image.open(filename)
        image = image.resize(size=(256, 256), resample=Image.NEAREST)
        image.save(f"{thumbnail_filename}", "jpeg")

        return thumbnail_filename

    def process_images(self):
        """
        Create a pool of processes to resize images.
        One process created per CPU
        """

        # Reserve one logical CPU core
        N_CPUS = psutil.cpu_count(logical=True) - 1

        with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
            
            # Get a list of files to process
            image_files = glob.glob("../data/train/*.jpeg")

            # Process the list of files, but split the work across the process pool to use all CPUs
            zip(image_files, executor.map(self.make_image_thumbnail, image_files))

    def create_directories(self, new_dir_name):
        """
        Create directories for images

        INPUT
            new_dir_name: str, name of directory to create

        OUTPUT
            Folder structure based on new_dir_name, with one folder per image class
        """

        if not os.path.exists(f"../data/train_resized/{new_dir_name}"):
                os.mkdir(f"../data/train_resized/{new_dir_name}")

        for val in list(range(0,5)):
            if not os.path.exists(f"../data/train_resized/{new_dir_name}/{val}"):
                os.mkdir(f"../data/train_resized/{new_dir_name}/{val}")

    def move_images(self, dict_images, new_dir_name):
        """
        Move images to folder based on label

        INPUT
            dict_images: dictionary of image name and label
                {"13_left": 0, "13_right": 1}
            new_dir_name: str, name of directory where images will be moved
        """

        # dict_images = dict(zip(
        #     self.df_labels["image"],
        #     self.df_labels["level"]
        #     ))

        # Move images to labeled directory
        for img in tqdm(dict_images.items()):
            shutil.move(f"../data/train_resized/{img[0]}.jpeg",
                f"../data/train_resized/{new_dir_name}/{img[1]}/{img[0]}.jpeg")


if __name__ == '__main__':
    
    preprocess = PreprocessImages()

    preprocess.process_images()

    # Create new directories for images
    preprocess.create_directories(new_dir_name="train")
    preprocess.create_directories(new_dir_name="valid")
    preprocess.create_directories(new_dir_name="test")

    X_train, X_valid, X_test = split_data()

    # Move images to respective directories
    preprocess.move_images(dict_images=X_train, new_dir_name="train")
    preprocess.move_images(dict_images=X_valid, new_dir_name="valid")
    preprocess.move_images(dict_images=X_test, new_dir_name="test")
