
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
import psutil
# from PIL import Image
from tqdm import tqdm


class PreprocessImages:

    def __init__(self):
        self.df_labels = pd.read_csv("../data/trainLabels.csv")

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
        image = cv2.imread(filename)
        image = cv2.resize(image, (256,256))
        cv2.imwrite(f"{thumbnail_filename}", image)
        # image = Image.open(filename)
        # image.thumbnail(size=(256, 256))
        # image.save(f"{thumbnail_filename}", "jpeg")

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

    def create_directories(self):
        """
        Create directories for images
        """

        for val in list(range(0,5)):
            if not os.path.exists(f"../data/train_resized/{val}"):
                os.mkdir(f"../data/train_resized/{val}")

    def move_images(self):
        """
        Move images to folder based on label
        """

        dict_images = dict(zip(
            self.df_labels["image"],
            self.df_labels["level"]
            ))

        # Move images to labeled directory
        for img in tqdm(dict_images.items()):
            shutil.move(f"../data/train_resized/{img[0]}.jpeg",
                f"../data/train_resized/{img[1]}/{img[0]}.jpeg")


if __name__ == '__main__':
    
    preprocess = PreprocessImages()

    preprocess.process_images()

    preprocess.create_directories()

    preprocess.move_images()
