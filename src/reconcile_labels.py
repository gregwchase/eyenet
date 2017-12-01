import os

import pandas as pd


def get_lst_images(file_path):
    """
    Reads in all files from file path into a list.

    INPUT
        file_path: specified file path containing the images.

    OUTPUT
        List of image strings
    """
    return [i for i in os.listdir(file_path) if i != '.DS_Store']


if __name__ == '__main__':
    trainLabels = pd.read_csv("../labels/trainLabels_master.csv")

    lst_imgs = get_lst_images('../data/train-resized-256/')

    new_trainLabels = pd.DataFrame({'image': lst_imgs})
    new_trainLabels['image2'] = new_trainLabels.image

    # Remove the suffix from the image names.
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    # Strip and add .jpeg back into file name
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(
        lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')

    # trainLabels = trainLabels[0:10]
    new_trainLabels.columns = ['train_image_name', 'image']

    trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
    trainLabels.drop(['black'], axis=1, inplace=True)
    # print(trainLabels.head(100))
    trainLabels = trainLabels.dropna()
    print(trainLabels.shape)

    print("Writing CSV")
    trainLabels.to_csv('../labels/trainLabels_master_256_v2.csv', index=False, header=True)
