from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import numpy as np


def get_dataset(dataset_name):

    path_train = f'data/tcls_datasets/{dataset_name}_train.csv'
    path_test = f'data/tcls_datasets/{dataset_name}_test.csv'

    df_train = pd.read_csv(path_train, index_col=[0])
    df_test = pd.read_csv(path_test, index_col=[0])

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

    # Create a smaller test dataset for faster training times
    df_test = pd.DataFrame(dataset['test']).sample(frac=1).reset_index(drop=True)
    per_class_test = np.round((instance_number*3) / df_test.label.nunique()).astype(int)
    df_test = df_test.groupby('label').head(per_class_test).sample(frac=1).reset_index(drop=True)

    small_dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test),
    })

    return small_dataset

def get_bio_dataset(dataset_name):

    path_train = f'bio/bio_data/cleaned/{dataset_name}.csv'

    df = pd.read_csv(path_train, index_col=[0])

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df),
        "test": Dataset.from_pandas(df)
    })

    return dataset

def get_small_bio_dataset(dataset_name, instance_number=200):
    '''
    arguments:
        instance_number: number of desired instances from original dataset for training data size.
    '''
    dataset = get_bio_dataset(dataset_name)

    # Create a smaller training dataset for faster training times
    df = pd.DataFrame(dataset['train']).sample(frac=1).reset_index(drop=True)
    df = df.head(instance_number).sample(frac=1).reset_index(drop=True)

    small_dataset = DatasetDict({
        "train": Dataset.from_pandas(df),
        "test": Dataset.from_pandas(df),
    })

    return small_dataset