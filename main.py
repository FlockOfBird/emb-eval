import llama_base
import bert_base
from data import *

'''
By running this file, a loop starts to generate embeddings based on all of the available models and datasets.
'''

# "llama_7B", "llama_13B", "llama_30B", "llama_65B", "llama2_7B", "llama2_13B", "llama2_70B"
models_llama = ["llama_7B", "llama2_7B"]
models_bert = ["bert"]

datasets = ["yelpp", "imdb", "agnews", "yelpf"]

for model in models_llama:
    for dataset_name in datasets:
        print('>>>>>>>>',model, dataset_name,'<<<<<<<<')
        llama_base.Llama_Embeddings(model, get_dataset(dataset_name), dataset_name)

for model in models_bert:
    for dataset_name in datasets:
        print('>>>>>>>>',model, dataset_name,'<<<<<<<<')
        dataset = get_dataset(dataset_name)
        bert_base.Bert_Embeddings(model, dataset, dataset_name)
