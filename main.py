from datasets import load_dataset
from src.preprocessing.preprocessing import preprocess


def main(): 
    # data = load_dataset("TeddyVDobreva/AML_project_dataset", split='train')
    # data = preprocess(data)
    # data.push_to_hub("TeddyVDobreva/AML_project_preprocessed_dataset", split='train')
    data = load_dataset("TeddyVDobreva/AML_project_preprocessed_dataset", split='train')


if __name__ == "__main__":
    main()