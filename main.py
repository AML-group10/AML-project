import json
import os
import subprocess

import numpy as np
from datasets import Dataset, load_dataset

# from src.preprocessing.preprocessing import preprocess
from PIL import Image

def prepare_dataset(dataset: Dataset, local_dir: str) -> None:
    """
    Downloads and prepares the dataset for training

    Args:
        dataset (Dataset): the dataset to be prepared
        local_dir (str): a directory to save the dataset
    """
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

def save_dataset_to_hub(path: str) -> None:
    """
    Saves a local dataset to the hub

    Args:
        path (str): local path to dataset
    """
    train_dataset = load_dataset("imagefolder", data_dir=(path + "train/"))
    valid_dataset = load_dataset("imagefolder", data_dir=(path + "valid/"))
    test_dataset = load_dataset("imagefolder", data_dir=(path + "test/"))
    train_dataset.push_to_hub("AML-group10/AML_project_preprocessed_dataset", "train")
    valid_dataset.push_to_hub("AML-group10/AML_project_preprocessed_dataset", "valid")
    test_dataset.push_to_hub("AML-group10/AML_project_preprocessed_dataset", "test")


def run_training(data_dir: str) -> None:
    """
    Runs a preprocessing and training loop

    Args:
        data_dir (str): a directory containing the dataset to train on
    """
    train_cmd = [
        "accelerate", "launch", "src/models/training/lora_training.py",
        "--pretrained_model_name_or_path", "segmind/tiny-sd",
        "--dataset_name", f"{data_dir}",
        "--output_dir", "AML-group10/lora-output",
        "--use_peft",
        "--lora_r", "4",
        "--lora_alpha", "4", 
        "--resolution", "256",
        "--train_batch_size", "512",
        "--num_train_epochs", "1",
        "--learning_rate", "1e-4",
        "--caption_column", "prompt",
        "--validation_prompt", "a man, smiling, wearing a suit and a tie, with curly hair",
        "--push_to_hub",
        "--dataset_config_name", "train",
        "--seed", "67",
        "--allow_tf32",
    ]

    subprocess.run(train_cmd, check=True)


def main():
    # data = load_dataset("AML-group10/AML_project_dataset", split='train')
    # data = preprocess(data)
    # data = split_data(data)
    # prepare_dataset(data["train"], "./faces/train/")
    # prepare_dataset(data["valid"], "./faces/valid/")
    # prepare_dataset(data["test"], "./faces/test/")

    # save_dataset_to_hub("./faces/")
    run_training("AML-group10/AML_project_preprocessed_dataset")


if __name__ == "__main__":
    main()
