import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    """
    Split data into training, validation, and test images
    """
    df_labels = pd.read_csv("../data/trainLabels.csv")

    # Create train/test data before correcting class imbalance
    # TODO: Stratify distributions
    X_train, X_valid = train_test_split(df_labels,
        test_size=0.2,
        random_state=42)

    X_train, X_test = train_test_split(X_train,
        test_size=len(X_valid),
        random_state=42)

    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_valid = pd.DataFrame(X_valid).reset_index(drop=True)
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    
    return X_train, X_valid, X_test

if __name__ == '__main__':

    X_train, X_valid, X_test = split_data()
