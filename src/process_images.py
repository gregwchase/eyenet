import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


class ProcessImages:

    def __init__(self):
        self.SCALE = 300
        self.SOURCE_DIR = "../data/sample/*.jpeg"
        self.TARGET_DIR = "../data/processed/"

    def scaleRadius(self, img):
        
        x=img[img.shape[0]//2,:,:].sum(1)
        
        r=(x>x.mean()/10).sum()/2
        
        s=self.SCALE * 1.0/r
        
        return cv2.resize(img,(0,0),fx=s,fy=s)

    def subtract_local_average_color(self, img):
        """
        Remove average local color per image
        Save image to new directory
        """
        
        try:
            a=cv2.imread(img)

            #scale img to a given radius
            a=self.scaleRadius(a)
            
            #subtract local mean color
            a=cv2.addWeighted(a,
                4,
                cv2.GaussianBlur(
                    a,(0,0),self.SCALE/30),
                -4,
                128)
            
            #remove outer 10%
            b=np.zeros(a.shape)
            cv2.circle(b,
                (a.shape[1]//2,a.shape[0]//2),
                int(self.SCALE * 0.9),(1,1,1),-1,8,0)
            
            a=a*b+128 * (1-b)
            
            print(f"Processed {img} successfully")
            
            # Original image name
            img_name = img.split("/")[-1]
            
            # Save processed image
            cv2.imwrite(f"{self.TARGET_DIR}{img_name}",a)
        
        except:
            print(f"Unable to process {img}")

    def create_directories(self, new_dir_name):
        """
        Create directories for images

        INPUT
            new_dir_name: str, name of directory to create

        OUTPUT
            Folder structure based on new_dir_name, with one folder per image class
        """

        if not os.path.exists(f"../data/processed/{new_dir_name}"):
                os.mkdir(f"../data/processed/{new_dir_name}")

        for val in list(range(0,5)):
            if not os.path.exists(f"../data/processed/{new_dir_name}/{val}"):
                os.mkdir(f"../data/processed/{new_dir_name}/{val}")

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
            shutil.move(f"../data/processed/{img[0]}.jpeg",
                f"../data/processed/{new_dir_name}/{img[1]}/{img[0]}.jpeg")

if __name__ == "__main__":

    preprocess = ProcessImages()

    IMG_FILES = glob.glob(preprocess.SOURCE_DIR)

    for i in tqdm(IMG_FILES):
        preprocess.subtract_local_average_color(img=i)

    # Create directories
    preprocess.create_directories(new_dir_name="train")
    preprocess.create_directories(new_dir_name="test")

    df_labels = pd.read_csv("../data/trainLabels.csv")

    # Create dictionary of image name and label
    dict_labels = df_labels.set_index("image").T.to_dict(orient="records")[0]

    # Move images to respective directories
    # preprocess.move_images(dict_images=X_train, new_dir_name="train")
    # preprocess.move_images(dict_images=X_test, new_dir_name="test")
