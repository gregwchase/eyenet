#!/bin/bash
# Download images for model

# Required libraries
# sudo apt install awscli
# pip install kaggle

# Download labels
kaggle competitions download -f trainLabels.csv.zip -c diabetic-retinopathy-detection

# Download training images
END=5
TRAIN_FILE=train.zip.00
for i in $(seq 1 $END);
    do
    FILE_NAME="${TRAIN_FILE}${i}"
    kaggle competitions download -f $FILE_NAME -c diabetic-retinopathy-detection;
    done

# Create directory for training data
mkdir train

# Move ZIP files to their directories
mv train.* train

# Extract data
7za x train/train.zip.001

# Move data to S3 (backup)
aws s3 mv train s3://$S3_BUCKET/train --recursive
