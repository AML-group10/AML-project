from datasets import Dataset

# tiny toy dataset
data = {
    "text": [
        "cat on a sofa",
        "dog in park",
        "naked person on beach",
        "a beautiful landscape",
        "topless model photo"
    ]
}

dataset = Dataset.from_dict(data)

print("Original dataset:")
print(dataset)

# FILTER: remove unsafe words
def filter_fn(example):
    banned = ["naked", "topless"]
    return not any(word in example["text"] for word in banned)

filtered = dataset.filter(filter_fn)

print("\nFiltered dataset:")
print(filtered)
print("\nRemaining texts:")
print(filtered["text"])