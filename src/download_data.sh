# sudo apt install awscli
# pip install kaggle-cli

kg download -u kaggle_user -p kaggle_password -c 'diabetic-retinopathy-detection'

# Create directories for test and train data
mkdir train
mkdir test

# Move ZIP files to their directories
mv train.* train
mv test.* test

# Extract data
7za x train.zip.001
7za x test.zip.001

# Move data to S3 (backup)
aws s3 mv train s3://gregwchase-diabetes/train --recursive
aws s3 mv test s3://gregwchase-diabetes/test --recursive
