from datasets import Dataset, DatasetDict
from src.CLIP_score import compute_CLIP


def preprocess(dataset: Dataset) -> Dataset:
    """
    Preprocesses the data in a dataset

    Args:
        dataset (Dataset): the dataset to be preprocessed

    Returns:
        Dataset: the preprocessed dataset
    """
    dataset = _remove_and_rename_features(dataset)
    dataset = _split_data(dataset)
    dataset = _remove_unwated_samples(dataset)
    dataset = _preprocess_captions(dataset)
    dataset = _preprocess_images(dataset)
    return dataset


def _remove_and_rename_features(dataset: Dataset) -> Dataset:
    """
    Removes the columns that will not be used in the training of the model and renames the existing columns with suitable names

    Args:
        dataset (Dataset): the dataset with the old columns

    Returns:
        Dataset: the dataset with removed columns and suitable names
    """
    dataset = dataset.remove_columns(["image_id", "image", "laion_caption", "sha256"])
    return dataset.rename_column("url", "image").rename_column(
        "human_caption", "prompt"
    )


def _split_data(dataset: Dataset) -> Dataset:
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


def _remove_unwated_samples(dataset: DatasetDict) -> DatasetDict:
    """
    Removes unwanted samples (null values, nudity, multiple people, low CLIP scores) from a dataset

    Args:
        dataset (Dataset): the dataset with all samples

    Returns:
        Dataset: the dataset with removed unwanted samples
    """
    # remove null values
    dataset = dataset.filter(lambda example: example["image"] is None)

    # remove nudity and multiple people
    dataset = dataset.filter(
        lambda example: any(
            [
                word in example["prompt"]
                for word in [
                    "naked",
                    "nude",
                    "breasts",
                    "undressed",
                    "topless",
                    "man and woman",
                    "men",
                    "women",
                    "people",
                ]
            ]
        )
    )

    # remove low CLIP scores
    dataset = dataset.filter(
        lambda example: compute_CLIP(example["image"], example["prompt"]) < 20
    )

    return dataset


def _preprocess_captions(dataset: Dataset) -> Dataset:
    """
    Preprocesses the captions in a dataset

    Args:
        dataset (Dataset): dataset to be preprocessed

    Returns:
        Dataset: dataset with preprocessed captions
    """
    return dataset.map(_preprocess_caption)


def _preprocess_images(dataset: Dataset) -> Dataset:
    """
    Preprocesses the images in a dataset

    Args:
        dataset (Dataset): dataset to be preprocessed

    Returns:
        Dataset: dataset with preprocessed images
    """
    # fixed resolution??
    # brightness normalization


def _preprocess_caption(example: dict) -> dict:
    """
    Preprocesses a single caption
    
    Args:
        example (dict): example with a caption

    Returns:
        dict: the same example with 
    """
    # lowercase
    caption = example['prompt'].lower()
    # cut out first 5 words
    caption = caption.split(' ', 6)[6]
    # set max length
    example['prompt'] = caption
    return example
    