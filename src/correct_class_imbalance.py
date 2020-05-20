import random

import pandas as pd


def split_data(df):
    """
    Split data into test and train data
    
    Function ensures patients aren't in both data sets
    """

    # Get list of unique patients
    lst_patients = df["patient_id"].unique().tolist()

    # Create list of unique test patients
    lst_test_patients = random.sample(lst_patients, k=N_TEST)

    # Subset test patients
    df_test = df[df["patient_id"].isin(lst_test_patients)]

    # Subset train patients
    df_train = df[~df["patient_id"].isin(lst_test_patients)]

    return df_train, df_test

def create_dict(df):
    """
    Create dictionary of values
    """
    dict_images = dict(zip(
            df["image"],
            df["level"]
            ))
    return dict_images

if __name__ == '__main__':

    df_labels = pd.read_csv("../data/trainLabels.csv")
    
    N_ROWS = len(df_labels)
    N_TEST = int(N_ROWS * 0.1)

    # Create patient ID
    df_labels["patient_id"] = df_labels["image"].str.split("_", expand=True)[0]

    df_train, df_test = split_data(df=df_labels)
