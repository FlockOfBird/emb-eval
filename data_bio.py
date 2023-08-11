def clean_smiles():
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


# def clean_drugdiscription():
#     Drug = pd.read_csv('data/drug_discription/drug_discription.csv')
#     Drug = Drug[Drug.Discription != ";;;;"]  # remove drugs without discription
#     Drug = Drug.reset_index(drop=True)

#     # dataset = load_dataset('csv', data_files = PATH)
#     for item in Drug['Discription']:
#         item = item.replace(';;;;', '')

#     PATH = 'data/drug_discription/drug_discription_cleaned.csv'
#     Drug.to_csv(PATH)

#     return


def clean_drugdiscription():
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