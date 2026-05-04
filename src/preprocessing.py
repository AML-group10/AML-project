from datasets import Dataset

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
    dataset = dataset.remove_columns(['image_id', 'image', 'laion_caption', 'sha256'])
    return dataset.rename_column('url', 'image').rename_column('human_caption', 'prompt')

def _split_data(dataset: Dataset) -> Dataset:
    """
    Spltits the data into train, validation, and test datasets

    Args:
        dataset (Dataset): the dataset to be split

    Returns:
        Dataset: a dataset split into train, validation, and test datasets
    """
    pass

def _remove_unwated_samples(dataset: Dataset) -> Dataset:
    """
    Removes unwanted samples (null values, nudity, multiple people, low CLIP scores) from a dataset

    Args:
        dataset (Dataset): the dataset with all samples

    Returns:
        Dataset: the dataset with removed unwanted samples
    """
    dataset = _remove_null_values(dataset)
    dataset = _remove_nudity(dataset)
    dataset = _remove_multiple_people_examples(dataset)
    dataset = _remove_low_CLIP(dataset)
    return dataset

def _preprocess_captions(dataset: Dataset) -> Dataset:
    """
    Preprocesses the captions in a dataset

    Args:
        dataset (Dataset): dataset to be preprocessed

    Returns:
        Dataset: dataset with preprocessed captions
    """
    # lowercase
    # tokenize
    # cut out tokens
    # set max length
    

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

def _remove_null_values(dataset: Dataset) -> Dataset:
    """
    Removes samples with null values for images from a dataset

    Args:
        dataset (Dataset): dataset with all samples

    Returns:
        Dataset: dataset with samples where images are not null
    """

def _remove_nudity(dataset: Dataset) -> Dataset:
    """
    Removes samples with nudity based on the caption from a dataset

    Args:
        dataset (Dataset): dataset with all samples

    Returns:
        Dataset: dataset with samples where images do not have nudity
    """

def _remove_multiple_people_examples(dataset: Dataset) -> Dataset:
    """
    Removes samples with multiple people based on the caption from a dataset

    Args:
        dataset (Dataset): dataset with all samples

    Returns:
        Dataset: dataset with samples where images have one person
    """

def _remove_low_CLIP(dataset: Dataset) -> Dataset:
    """
    Removes samples with low prompt to image correlation from a dataset

    Args:
        dataset (Dataset): dataset with all samples

    Returns:
        Dataset: dataset with samples where prompts and images have higher correlation
    """
