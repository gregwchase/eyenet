import pandas as pd
import os
import numpy as np

if __name__ == '__main__':

    trainLabels = pd.read_csv("../labels/trainLabels_master.csv")
    # trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')

    lst_imgs = [i for i in os.listdir('../data/sample-resized-256') if i != '.DS_Store']

    new_trainLabels = pd.DataFrame({'image': lst_imgs})
    # new_trainLabels['level'] = np.nan
    new_trainLabels['image2'] = new_trainLabels.image

    # Remove the suffix from the image names.
    new_trainLabels['image2'] = new_trainLabels.loc[:,'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    # Strip and add .jpeg back in
    new_trainLabels['image2'] = new_trainLabels.loc[:,'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')

    trainLabels = trainLabels[0:10]
    new_trainLabels.columns = ['train_image_name', 'image']

    # print(trainLabels.head())
    # print(new_trainLabels.head(20))

    trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
    trainLabels.drop(['black'], axis = 1, inplace=True)
    print(trainLabels)
