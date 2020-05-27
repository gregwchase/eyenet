import cv2
import glob
import numpy as np

from concurrent.futures import ProcessPoolExecutor
import psutil


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

if __name__ == "__main__":

    preprocess = ProcessImages()

    IMG_FILES = glob.glob(preprocess.SOURCE_DIR)

    for i in IMG_FILES:
        preprocess.subtract_local_average_color(img=i)
    
    # # # Reserve one logical CPU core
    # N_CPUS = psutil.cpu_count(logical=True) - 1

    # with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        
    #     # Get a list of files to process
    #     image_files = glob.glob(preprocess.SOURCE_DIR)

    #     # Process the list of files, but split the work across the process pool to use all CPUs
    #     zip(image_files, executor.map(preprocess.subtract_local_average_color, image_files))