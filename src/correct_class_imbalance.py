from sklearn.preprocessing import LabelEncoder
import pandas as pd
from itertools import chain
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df_labels = pd.read_csv("../data/trainLabels.csv")

    # Create train/test data before correcting class imbalance
    # TODO: Remove duplicate images from test set
    # TODO: Make distribution similar to test set?
    X_train, X_test = train_test_split(df_labels,
        test_size=0.2,
        random_state=42)

    # Create Label Encoder
    le = LabelEncoder()

    le.fit(X_train["image"])

    X_transform = le.transform(X_train["image"])
    
    """
    Over-sample image data
    """
    ros = RandomOverSampler()

    X_resampled, Y_resampled = ros.fit_resample(X_transform.reshape(-1,1), X_train["level"])

    df_balanced_classes = pd.DataFrame({"image": list(chain(*X_resampled)), "level": Y_resampled})
        
    # Get image_name from inverse transformation
    df_balanced_classes["image_name"] = le.inverse_transform(df_balanced_classes["image"])
