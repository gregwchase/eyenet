# EyeNet

## Improved Treatment of Diabetic Retinopathy Using Deep Learning

## Objective
[What is Diabetic Retinopathy?](http://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/basics/definition/con-20023311)

Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people.

The need for a comprehensive and automated method of DR screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With photos of eyes as input, the goal of this capstone is to create a new model, ideally resulting in realistic clinical potential.


## Motivations for Project
`"People don't know what they want until you've shown them." - Steve Jobs`

* I've been interested in image classification since the cohort began, and doing it on a large scale. This project is of interest due to how many people are affected, and how many people could seek treatment at an earlier stage.


## Data

The data is originally from a 2015 Kaggle competition. However, this data is much different.

In most Kaggle competitions, the data is very clean, giving the data scientist very little to preprocess. With this dataset, this isn't the case.

All images are taken of different people, using different cameras, and of different sizes. Pertaining the preprocessing section, this data is extremely noisy, and requires lots of preprocessing to get all images to a useable size for training a model.

## Exploratory Data Analysis

The very first item analyzed was the training labels. While there are
five categories to predict against, the plot below shows the severe class imbalance
in the original dataset.

![EDA - Class Imbalance](images/eda/Retinopathy_vs_Frequency_All.png)

Due to the class imbalance, steps were taken in preprocessing in order to rectify the imbalance, and when training the model.

## Preprocessing

The preprocessing pipeline is the following:

1. Download all images to EC2 using the [download script](src/download_data.sh).
2. Crop all images
3. Zoom in on images
4. Rotate images (pertaining to class imbalance)
5. Resize images to a uniform size, using the [rotation script](src/resize_images.py).
5. Convert all images to array of NumPy arrays, using the [conversion script](src/image_to_array.py).


## References

1. [Diabetic Retinopathy Winners' Interview: 4th place, Julian & Daniel](http://blog.kaggle.com/2015/08/14/diabetic-retinopathy-winners-interview-4th-place-julian-daniel/)
