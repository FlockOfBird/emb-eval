import llama_base
from data import *

'''
By running this file, a loop starts to generate embeddings based on all of the available models and datasets.
'''

# "llama_7B", "llama_13B", "llama_30B", "llama_65B", "llama2_7B", "llama2_13B", "llama2_70B"
models = ["llama_7B", "llama2_7B"]

datasets = ["imdb", "agnews", "yelp_review_polarity", "yelp_review_full"]
datasets_small = ["imdb_small", "agnews_small", "small_yelp_review_polarity", "small_yelp_review_full"]
instance_number = 1000
dataset_getters = {
    "imdb": get_imdb(),
    "agnews": get_agnews(),
    "yelp_review_polarity": get_yelpp(),
    "yelp_review_full": get_yelpf(),

    "imdb_small": get_small_imdb(instance_number),
    "agnews_small": get_small_agnews(instance_number),
    "small_yelp_review_polarity": get_small_yelpp(instance_number),
    "small_yelp_review_full": get_small_yelpf(instance_number)
}

for model in models:
    for dataset_name in datasets_small:
        print('>>>>>>>>',model, dataset_name,'<<<<<<<<')
        llama_base.Llama_Embeddings(model, dataset_getters[dataset_name], dataset_name)
