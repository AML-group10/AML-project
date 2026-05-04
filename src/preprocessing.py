

def preprocess(dataset):
    dataset = _remove_and_rename_features(dataset)

def _remove_and_rename_features(dataset):
    dataset = dataset.remove_columns(['image_id', 'image', 'laion_caption', 'sha256'])
    return dataset.rename_column('url', 'image')