import pandas as pd
import numpy as np
import os
import glob

def clean(file):
    print(file)
    df = pd.read_csv(file, sep=',')
    print(df.head())

    # df = df.drop(["Drug ID", "Drug Name", "Unnamed: 0"], axis=1)
    df = df["SMILES"]
    # df = df.sort_index(axis=1, ascending=False)
    print(df.head())

    splited_path = file.split("/")
    data_name = splited_path[-1]
    print(splited_path[-1])
    df.to_csv(f'./bio_data/cleaned/{data_name}', index=False, sep='\t')

file = './bio_data/drug_discription/structure_links.csv'
clean(file)

# for file in glob.iglob('./data/bio/*.csv'):
#     clean(file)
