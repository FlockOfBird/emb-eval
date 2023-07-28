from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd


def get_sample_data():
    '''
        returns:
            a hugging face dataset containing test and train dataset
    '''
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
    '''
        This file consists of 25,000 training and 25,000 testing samples of imdb reviews that contain 2 columns. 
        The first column is text, the second column is label. 
        The class ids are numbered 0/1. 0 represents negative review and 1 represents positive review.

        returns:
            a hugging face dataset containing test and train dataset
    '''
    dataset = load_dataset('imdb')

    return dataset["train"], dataset["test"]


def get_small_imdb(instance_number):
    '''
        arguments:
            instance_number: number of desired instances from original imdb for train. test size will be n/10 of this number.

        returns:
            a hugging face dataset containing test and train dataset
    '''
    dataset_train, dataset_test = get_imdb()
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_test.shuffle(seed=42).select(
        [i for i in list(range(int(int(instance_number/10))))])

    return small_train_dataset, small_test_dataset


def get_smiles():
    '''
        This file consists of 11,583 training and 0 testing samples of drugIDs and their SMILES code that contain 3 columns. 
        The first column is Unnamed which enumerates data, the second column is DrugBankID and third column is SMILES representation. 
        This data is unlabeled for unsupervised tasks.
        
        returns:
            a hugging face dataset containing test and train dataset
    '''
    PATH = 'data/smiles/drugID_smiles.csv'
    dataset = load_dataset('csv', data_files=PATH)
    dataset["test"] = None
    return dataset["train"], dataset["test"]


def get_small_smiles(instance_number):
    '''
        arguments:
            instance_number: number of desired instances from original smiles for train. test size will be n/10 of this number.

        returns:
            a hugging face dataset containing test and train dataset
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
    '''
        This file consists of 8,723 training and 0 testing samples of drugIDs and their SMILES code that contain 4 columns. 
        The first column is Unnamed which enumerates data, the second column is DrugID, the third column is Drug_Name, and fourth column is Discription. 
        This data is unlabeled for unsupervised tasks.

        returns:
            a hugging face dataset containing test and train dataset
    '''
    PATH = 'data/drug_discription/drug_discription_cleaned.csv'
    dataset = load_dataset('csv', data_files=PATH)
    dataset["test"] = None
    return dataset["train"], dataset["test"]


def get_small_drugdiscription(instance_number):
    '''
        arguments:
            instance_number: number of desired instances from original drugdiscription for train. test size will be n/10 of this number.
    '''
    dataset_train = get_drugdiscription()
    print(dataset_train, instance_number)
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number/10)))])

    return small_train_dataset, small_test_dataset


def get_agnews(concat_title=False):
    '''
        This file consists of 120,000 training and 7,600 testing samples of news articles that contain 3 columns. 
        The first column is Class Id, the second column is Title and the third column is Description. 
        The class ids are numbered 1-4. 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.

        arguments:
            concat_title: if concat title and description text with each other

        returns:
            a hugging face dataset containing test and train dataset
    '''
    PATH_TRAIN = 'data/agnews_dataset/train.csv'
    PATH_TEST = 'data/agnews_dataset/test.csv'

    df_train = pd.read_csv(PATH_TRAIN)
    df_test = pd.read_csv(PATH_TEST)

    df_train = df_train.rename(columns={"Description": "text", "Class Index": "label"})
    df_test = df_test.rename(columns={"Description": "text", "Class Index": "label"})

    if concat_title:
        df_train["text"] = df_train["Title"] + '. ' + df_train["text"]
        df_train = df_train.drop(["Title"], axis=1)
        df_test["text"] = df_test["Title"] + '. ' + df_test["text"]
        df_test = df_test.drop(["Title"], axis=1)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test)
    })

    return dataset["train"], dataset["test"]

def get_small_agnews(instance_number):
    '''
    arguments:
        instance_number: number of desired instances from original agnews for train. test size will be n/10 of this number.
    '''
    dataset_train, dataset_test = get_agnews()
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_test.shuffle(seed=42).select(
        [i for i in list(range(int(int(instance_number/10))))])

    return small_train_dataset, small_test_dataset

def get_yelpp():
    '''
        This file consists of 560,000 training and 38,000 testing samples of news articles that contain 3 columns. 
        The first column is label, the second column is text. 
        Negative polarity is class 1, and positive class 2.

        returns:
            a hugging face dataset containing test and train dataset
    '''
    PATH_TRAIN = 'data/yelp_review_polarity/train.csv'
    PATH_TEST = 'data/yelp_review_polarity/test.csv'

    df_train = pd.read_csv(PATH_TRAIN)
    df_train.columns = ["label", "text"]

    df_test = pd.read_csv(PATH_TEST)
    df_test.columns = ["label", "text"]

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test)
    })

    return dataset["train"], dataset["test"]

def get_small_yelpp(instance_number):
    '''
        arguments:
            instance_number: number of desired instances from original agnews for train. test size will be n/10 of this number.
    '''
    dataset_train, dataset_test = get_yelpp()
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_test.shuffle(seed=42).select(
        [i for i in list(range(int(int(instance_number/10))))])

    return small_train_dataset, small_test_dataset

def get_yelpf():
    '''
        This file consists of 650,000 training and 50,000 testing samples of reviews that contain 2 columns. 
        The first column is label with 5 uinique as stars given to each resturant, the second column is text. 
        
        returns:
            a hugging face dataset containing test and train dataset
    '''
    PATH_TRAIN = 'data/yelp_review_full/train.csv'
    PATH_TEST = 'data/yelp_review_full/test.csv'

    df_train = pd.read_csv(PATH_TRAIN)
    df_train.columns = ["label", "text"]

    df_test = pd.read_csv(PATH_TEST)
    df_test.columns = ["label", "text"]

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test)
    })

    return dataset["train"], dataset["test"]

def get_small_yelpf(instance_number):
    '''
        arguments:
            instance_number: number of desired instances from original agnews for train. test size will be n/10 of this number.
    '''
    dataset_train, dataset_test = get_yelpf()
    # Create a smaller training dataset for faster training times
    small_train_dataset = dataset_train.shuffle(seed=42).select(
        [i for i in list(range(int(instance_number)))])
    small_test_dataset = dataset_test.shuffle(seed=42).select(
        [i for i in list(range(int(int(instance_number/10))))])

    return small_train_dataset, small_test_dataset

if __name__ == "__main__":
    data_train, data_test = get_yelpf() # change dataset name to test functions
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    print(len(df_train))
    print(len(df_test))
    print(df_train.head())


