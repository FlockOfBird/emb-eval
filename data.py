from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import numpy as np


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


def get_small_dataset(dataset_name, instance_number=1500):
    '''
    arguments:
        instance_number: number of desired instances from original dataset for training data size.
    '''
    dataset = get_dataset(dataset_name)

    # Create a smaller training dataset for faster training times
    df_train = pd.DataFrame(dataset['train']).sample(frac=1).reset_index(drop=True)
    per_class_train = np.round(instance_number / df_train.label.nunique()).astype(int)
    df_train = df_train.groupby('label').head(per_class_train).sample(frac=1).reset_index(drop=True)

    small_dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": dataset['test']
    })

    return small_dataset
