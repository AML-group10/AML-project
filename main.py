from datasets import load_dataset
from src.preprocessing import preprocess


def main(): 
    data = load_dataset("TeddyVDobreva/AML_project_dataset", split='train[:20]')
    preprocess(data)




if __name__ == "__main__":
    main()
