import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    """
    Split data into training, validation, and test images
    """
    df_labels = pd.read_csv("../data/trainLabels.csv")

    # Create train/test data before correcting class imbalance
    # TODO: Stratify distributions

    # Create validation data (20% of whole dataset)
    X_valid = df_labels.sample(frac=0.2)
    df_labels.drop(X_valid.index, inplace=True)
    X_valid = create_dict(df=X_valid)

    # Create test data (20% of whole dataset)
    X_test = df_labels.sample(n=len(X_valid))
    df_labels.drop(X_test.index, inplace=True)
    X_test = create_dict(df=X_test)

    # Create training data
    X_train = df_labels
    X_train = create_dict(df=X_train)

    return X_train, X_valid, X_test

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

    X_train, X_valid, X_test = split_data()
