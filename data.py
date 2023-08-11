from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

def get_dataset(dataset_name):

    path_train = f'data/tcls_datasets/{dataset_name}_train.csv'
    path_test = f'data/tcls_datasets/{dataset_name}_test.csv'

    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test)
    })

    return dataset

def get_small_dataset(dataset_name, instance_number=1000):
    '''
    arguments:
        instance_number: number of desired instances from original agnews for train. test size will be n/10 of this number.
    '''
    dataset = get_dataset(dataset_name)
    
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset["train"].shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset["test"].shuffle(seed=42).select(
        [i for i in list(range(int(instance_number/10)))])

    dataset = DatasetDict({
        "train": small_train_dataset,
        "test": small_test_dataset
    })

    return dataset

