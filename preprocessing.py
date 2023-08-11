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


def clean_imdb():
    '''
        This file consists of 25,000 training and 25,000 testing samples of imdb reviews that contain 2 columns. 
        The first column is text, the second column is label. 
        The class ids are numbered 0/1. 0 represents negative review and 1 represents positive review.
    '''
    dataset = load_dataset('imdb')
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    print(df_train)

    df_train.to_csv('data/tcls_datasets/imdb_train.csv')
    df_test.to_csv('data/tcls_datasets/imdb_test.csv')

    return df_train, df_test

def clean_agnews(concat_title=True):
    '''
        This file consists of 120,000 training and 7,600 testing samples of news articles that contain 3 columns. 
        The first column is Class Id, the second column is Title and the third column is Description. 
        The class ids are numbered 1-4. 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.

        arguments:
            concat_title: if concat title and description text with each other
        
        returns:
            two dataframes consisting of train and test data
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

    df_train.to_csv('data/tcls_datasets/agnews_train.csv')
    df_test.to_csv('data/tcls_datasets/agnews_test.csv')

    return df_train, df_test

def clean_yelpp():
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

    df_train.to_csv('data/tcls_datasets/yelpp_train.csv')
    df_test.to_csv('data/tcls_datasets/yelpp_test.csv')

def clean_yelpf():
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

    df_train.to_csv('data/tcls_datasets/yelpf_train.csv')
    df_test.to_csv('data/tcls_datasets/yelpf_test.csv')

if __name__ == "__main__":
    clean_small_imdb(1000) # change dataset name to test functions


