import pandas as pd
from sklearn.model_selection import train_test_split


def create_subject_splits(metadata_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    df = pd.read_csv(metadata_path)

    subject_ids = df["id"].unique()

    train_subjects, temp_subjects = train_test_split(
        subject_ids,
        test_size=(1 - train_ratio),
        random_state=random_state,
        shuffle=True
    )

    val_size_adjusted = val_ratio / (val_ratio + test_ratio)

    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=(1 - val_size_adjusted),
        random_state=random_state,
        shuffle=True
    )

    return {
        "train": set(train_subjects),
        "val": set(val_subjects),
        "test": set(test_subjects)
    }