import llama_base
import bert_base
from data import *
'''
By running this file, a loop starts to generate embeddings based on all of the available models and datasets.
'''
'''
options for models_llama:
    "llama-7B", "llama-13B", "llama-30B", "llama-65B", "llama2-7B", "llama2-13B", "llama2-70B"
'''
models_llama = ["llama2-7B"]
'''
options for models_bert:
    "bert"
'''
models_bert = ["bert"]

'''
options for datasets:
    "yelpp", "imdb", "agnews", "yelpf"
'''
datasets = ["yelpf"] #, "yelpp", "imdb", "agnews"

# for model in models_llama:
#     llama_base.Llama_Embeddings(model, datasets)

# for model in models_bert:
#     bert_base.Bert_Embeddings(model, datasets)

'''
options for bio datasets:
    "hiv", "bace", "clintox", "structure_links", "drug_discription"
'''
bio_datasets = ["structure_links", "drug_discription"]
for model in models_llama:
    llama_base.Llama_Embeddings(model, bio_datasets)
