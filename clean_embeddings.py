import pandas as pd
import os
import glob

for file in glob.iglob('./results/embeddings/*.csv'):
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