import pandas as pd
import numpy as np
import os
import glob

def sep_vecs_txtlbl(file):
    print(file)
    df = pd.read_csv(file, sep='\t')
    # df = df.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1)
    # df = df.sort_index(axis=1, ascending=False)
    vecs = df.iloc[:,2:]
    txt_labels = df.iloc[:,0:2]

    cols = vecs.columns.values.tolist()
    cols = [eval(i) for i in cols]
    cols.sort()
    cols = map(str, cols)
    vecs = vecs[cols]

    splited_path = file.split("/")

    vecs_path = splited_path[3].replace('embeddings', 'vecs')
    vecs.to_csv(f'./results/embeddings_cleaned/{vecs_path}', index=False, sep='\t')

    txt_labels_path = splited_path[3].replace('embeddings', 'txtlbl')
    txt_labels.to_csv(f'./results/embeddings_cleaned/{txt_labels_path}', index=False, sep='\t')

def sort_vecs(file):
    print(file)
    df = pd.read_csv(file, sep='\t')
    vecs = df.iloc[:,2:]
    txt_labels = df.iloc[:,0:2]

    cols = vecs.columns.values.tolist()
    cols = [eval(i) for i in cols]
    cols.sort()
    cols = map(str, cols)
    vecs = vecs[cols]
    df = pd.concat([txt_labels, vecs], axis=1)
    print(df.head())

    splited_path = file.split("/")

    df_path = splited_path[3]
    df.to_csv(f'./results/embeddings_cleaned/{df_path}', index=False, sep='\t')


def select_n_instances(file, n):
    df = pd.read_csv(file, sep='\t')
    print(file)
    print(df)
    per_class = np.round(n / df.label.nunique()).astype(int)
    print(per_class)
    df = df.groupby('label').head(per_class).sample(frac=1).reset_index(drop=True)

    splited_path = file.split("/")

    df_path = splited_path[3]
    df.to_csv(f'./results/embeddings/{df_path}', index=False, sep='\t')
    print(df)

def join_vecs_txtlbl(file_txtlbl, file_vecs):
    df_txtlbl = pd.read_csv(file_txtlbl, sep='\t')
    print(df_txtlbl)
    df_vecs = pd.read_csv(file_vecs, sep='\t')
    print(df_vecs)

    df = pd.concat([df_txtlbl, df_vecs], axis=1)
    df.to_csv(f'./results/embeddings/llama2-7B_yelpf_test_embeddings.csv', index=False, sep='\t')
    print(df)

for file in glob.iglob('./results/embeddings/*.csv'):
    sep_vecs_txtlbl(file)
    # sep_vecs_txtlbl(file)
    # sort_vecs(file)
    # select_n_instances(file, 4500)
    # join_vecs_txtlbl(file_txtlbl, file_vecs)