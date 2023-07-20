from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd


def get_sample_data():
    dataset_train = {
        'text': [
            'I hate cats',
            'The sky is blue',
            'OpenAI is amazing',
            'The sun is shining',
            'Pizza tastes awful',
            'I do not like reading books',
            'The cat is sleeping',
            'Music brings joy',
            'Coding is the worst',
            'The dog is barking'
        ],
        'label': [0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    }
    dataset_test = {
        'text': [
            'I feel sick',
            'The movie is beautiful'
        ],
        'label': [0, 1]
    }
    df_train = pd.DataFrame(dataset_train)
    df_test = pd.DataFrame(dataset_test)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test)
    })

    return dataset["train"], dataset["test"]


def get_imdb():
    dataset = load_dataset('imdb')

    return dataset["train"], dataset["test"]


def get_small_imdb(instance_number):
    '''
        instance_number: number of desired instances from original imdb for train. test size will be n/10 of this number.
    '''
    dataset_train, dataset_test = get_imdb()
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_test.shuffle(seed=42).select(
        [i for i in list(range(int(int(instance_number/10))))])

    return small_train_dataset, small_test_dataset


def get_smiles():
    PATH = 'data/smiles/drugID_smiles.csv'
    dataset = load_dataset('csv', data_files=PATH)
    dataset["test"] = None
    return dataset["train"], dataset["test"]


def get_small_smiles(instance_number):
    '''
        instance_number: number of desired instances from original imdb for train. test size will be n/10 of this number.
    '''
    dataset_train, dataset_test = get_smiles()
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number/10)))])

    return small_train_dataset, small_test_dataset


def clean_drugdiscription():
    Drug = pd.read_csv('data/drug_discription/drug_discription.csv')
    Drug = Drug[Drug.Discription != ";;;;"]  # remove drugs without discription
    Drug = Drug.reset_index(drop=True)

    # dataset = load_dataset('csv', data_files = PATH)
    for item in Drug['Discription']:
        item = item.replace(';;;;', '')

    PATH = 'data/drug_discription/drug_discription_cleaned.csv'
    Drug.to_csv(PATH)

    return


def get_drugdiscription():
    PATH = 'data/drug_discription/drug_discription_cleaned.csv'
    dataset = load_dataset('csv', data_files=PATH)
    dataset["test"] = None
    return dataset["train"], dataset["test"]


def get_small_drugdiscription(instance_number):
    '''
        instance_number: number of desired instances from original imdb for train. test size will be n/10 of this number.
    '''
    dataset_train = get_drugdiscription()
    print(dataset_train, instance_number)
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number/10)))])

    return small_train_dataset, small_test_dataset
