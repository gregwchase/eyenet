import cv2
import glob
import numpy as np


class ProcessImages:

    def __init__(self):
        self.SCALE = 300
        self.IMG_DIR = "../data/sample/*.jpeg"

    def scaleRadius(self, img):
        
        x=img[img.shape[0]//2,:,:].sum(1)
        
        r=(x>x.mean()/10).sum()/2
        
        s=self.SCALE * 1.0/r
        
        return cv2.resize(img,(0,0),fx=s,fy=s)

    def subtract_local_average_color(self):
        """
        Remove average local color per image
        Save image to new directory
        """
        for f in glob.glob(self.IMG_DIR):
            
            try:
                a=cv2.imread(f)

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
                
                print(f"Processed {f} successfully")
                
                # Original image name
                img_name = f.split("/")[-1]
                
                # Save processed image
                cv2.imwrite(f"../data/processed/{img_name}",a)
            
            except:
                print(f"Unable to process {f}")

if __name__ == "__main__":

    preprocess = ProcessImages()

    preprocess.subtract_local_average_color()
    