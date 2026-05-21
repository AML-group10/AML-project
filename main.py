import json
import os

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# from src.preprocessing.preprocessing import preprocess
from PIL import Image


def split_data(dataset: Dataset) -> Dataset:
    """
    Spltits the data into train, validation, and test datasets

    Args:
        dataset (Dataset): the dataset to be split

    Returns:
        Dataset: a dataset split into train, validation, and test datasets
    """
    # 70% train, 30% test + validation
    train_valtest = dataset.train_test_split(test_size=0.3, seed=42)
    # Split the 30% test in half for validation and half for testing
    val_test = train_valtest["test"].train_test_split(test_size=0.5, shuffle=False)

    train_test_valid_dataset = DatasetDict(
        {
            "train": train_valtest["train"],
            "valid": val_test["train"],
            "test": val_test["test"],
        }
    )
    return train_test_valid_dataset


def prepare_dataset(dataset: DatasetDict):
    local_dir = "./faces/"
    os.makedirs(local_dir, exist_ok=True)
    with open(f"{local_dir}metadata.jsonl", "w") as outfile:
        for index in range(len(dataset)):
            img = Image.fromarray(np.array(dataset[index]["image"], dtype="uint8"))
            img.save(f"{local_dir}image{index}.jpeg")
            entry = {
                "file_name": f"image{index}.jpeg",
                "prompt": dataset[index]["prompt"],
            }
            json.dump(entry, outfile)
            outfile.write("\n")


def main():
    # data = load_dataset("TeddyVDobreva/AML_project_dataset", split='train')
    # data = preprocess(data)
    # data.push_to_hub("TeddyVDobreva/AML_project_preprocessed_dataset", split='train')
    data = load_dataset("TeddyVDobreva/AML_project_preprocessed_dataset", split="train")
    data = split_data(data)
    prepare_dataset(data["train"])
    print(data)


if __name__ == "__main__":
    main()
