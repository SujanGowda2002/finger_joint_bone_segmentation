import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import HandOADataset
from src.data.preprocessing import get_transforms
from src.data.subject_split import create_subject_splits


def summarize_dataset(dataset, split_name):
    joint_counter = Counter()
    label_counter = Counter()
    subject_ids = set()
    missing_labels = 0

    for i in range(len(dataset)):
        sample = dataset[i]

        joint = sample["joint"]
        label = sample["label"].item()
        subject_id = sample["subject_id"]

        joint_counter[joint] += 1
        subject_ids.add(subject_id)

        if label != label:  # checks NaN
            missing_labels += 1
        else:
            label_counter[int(label)] += 1

    print(f"\n===== {split_name.upper()} DATASET SUMMARY =====")
    print(f"Total image samples: {len(dataset)}")
    print(f"Unique subjects: {len(subject_ids)}")
    print(f"Missing labels: {missing_labels}")

    print("\nImages per joint:")
    for joint, count in sorted(joint_counter.items()):
        print(f"  {joint}: {count}")

    print("\nKL label distribution:")
    for label, count in sorted(label_counter.items()):
        print(f"  KL={label}: {count}")


def summarize_all_images(image_dir):
    image_files = sorted(os.listdir(image_dir))
    joint_counter = Counter()
    subject_ids = set()

    for image_name in image_files:
        if not os.path.isfile(os.path.join(image_dir, image_name)):
            continue

        if "_" not in image_name:
            continue

        parts = image_name.split("_")
        if len(parts) < 2:
            continue

        try:
            subject_id = int(parts[0])
            joint = parts[1].split(".")[0].upper()
        except ValueError:
            continue

        subject_ids.add(subject_id)
        joint_counter[joint] += 1

    print("===== OVERALL IMAGE FOLDER SUMMARY =====")
    print(f"Total image files: {len(image_files)}")
    print(f"Unique subjects in image folder: {len(subject_ids)}")

    print("\nOverall images per joint:")
    for joint, count in sorted(joint_counter.items()):
        print(f"  {joint}: {count}")


def main():
    image_dir = "data/raw/images"
    metadata_path = "data/raw/Hand.csv"

    print("Creating subject splits...")
    splits = create_subject_splits(metadata_path)

    print("\nChecking overall image folder...")
    summarize_all_images(image_dir)

    transform = get_transforms()

    train_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=transform,
        allowed_subjects=splits["train"]
    )

    val_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=transform,
        allowed_subjects=splits["val"]
    )

    test_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=transform,
        allowed_subjects=splits["test"]
    )

    summarize_dataset(train_dataset, "train")
    summarize_dataset(val_dataset, "validation")
    summarize_dataset(test_dataset, "test")


if __name__ == "__main__":
    main()