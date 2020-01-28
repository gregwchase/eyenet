from sklearn.preprocessing import LabelEncoder
import pandas as pd
from itertools import chain
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df_labels = pd.read_csv("../data/trainLabels.csv")

    le = LabelEncoder()

    le.fit(df_labels["image"])

    X = le.transform(df_labels["image"])

    ros = RandomOverSampler()

    X_resampled, Y_resampled = ros.fit_resample(X.reshape(-1,1), df_labels["level"])

    df_balanced_classes = pd.DataFrame({"image": list(chain(*X_resampled)), "level": Y_resampled})

    # Get name of image from inverse transformation
    # df_balanced_classes["image"] = le.inverse_transform(df_balanced_classes["image"])

    # Create test/train split
    # TODO: Remove duplicate images from test set
    # TODO: Make distribution similar to test set?
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_balanced_classes["image"],
        df_balanced_classes["level"],
        test_size=0.2,
        random_state=42)