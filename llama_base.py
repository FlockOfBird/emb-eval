import logging
import json
from tqdm import tqdm

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from accelerate import infer_auto_device_map

import tensorflow as tf
import numpy as np
import pandas as pd

# from data import *

class Llama_Embeddings:
    def __init__(self, model_name, dataset_getter, dataset_name):

        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info(f'loading dataset')
        self.dataset_name = dataset_name
        self.train_data, self.test_data = dataset_getter

        logging.info('loading model and tokenizer')
        self.model_name = model_name
        ##! generate an error if the model name is not a valid key
        models_path = {
            "llama_7B": "./llama_converted/7B",
            "llama_13B": "./llama_converted/13B",
            "llama_30B": "./llama_converted/30B",
            "llama_65B": "./llama_converted/65B",
            "llama2_7B": "./llama2_converted/7B"
        }
        PATH_TO_CONVERTED_WEIGHTS = models_path[model_name]

        # Set device to auto to utilize GPU
        device = "auto" # balanced_low_0, auto, balanced, sequential

        if model_name == "llama_30B":
            print("loading llama 30B takes much longer time due to GPU management issues.")
            self.model = LlamaForCausalLM.from_pretrained(
                PATH_TO_CONVERTED_WEIGHTS,
                device_map=device,
                max_memory={0: "12GiB", 1: "12GiB", 2:"12GiB", 3:"12GiB"},
                offload_folder="offload"
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                PATH_TO_CONVERTED_WEIGHTS,
                device_map=device
            )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            PATH_TO_CONVERTED_WEIGHTS
        )

        embeddings_train, embeddings_test = self.get_embeddings(save_embeddings=True)


    def get_embeddings(self, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            arguments:
                save_embeddings: T/F value to save the model in results directory
        '''
        embed_train_data = True
        embed_test_data = True

        if(embed_train_data):
            embeddings_train = []
            for data_row in tqdm(self.train_data):
                tokens = self.tokenizer(data_row['text'])
                input_ids = tokens['input_ids']

                # Obtain sentence embedding
                with torch.no_grad():
                    input_embeddings = self.model.get_input_embeddings()
                    embedding = input_embeddings(
                        torch.LongTensor([input_ids]))
                    embedding = torch.mean(
                        embedding[0], 0).cpu().detach()

                    embeddings_train.append(embedding)


        if(embed_test_data):
            embeddings_test = []
            for data_row in tqdm(self.test_data):
                tokens = self.tokenizer(data_row['text'])
                input_ids = tokens['input_ids']

                # Obtain sentence embedding
                with torch.no_grad():
                    input_embeddings = self.model.get_input_embeddings()
                    embedding = input_embeddings(
                        torch.LongTensor([input_ids]))
                    embedding = torch.mean(
                        embedding[0], 0).cpu().detach()
                    embeddings_test.append(embedding)

        if save_embeddings:
            logging.info('saving train data embeddings')
            embeddings_train = np.array(embeddings_train)
            train_data = pd.concat([pd.DataFrame(self.train_data), pd.DataFrame(embeddings_train)], axis=1)
            train_data.to_csv(f'results/embeddings/{self.model_name}_base_{self.dataset_name}_embeddings_train.csv', sep='\t')

            logging.info('saving test data embeddings')
            embeddings_test = np.array(embeddings_test)
            test_data = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(embeddings_test)], axis=1)
            test_data.to_csv(f'results/embeddings/{self.model_name}_base_{self.dataset_name}_embeddings_test.csv', sep='\t')

        return embeddings_train, embeddings_test

# if __name__ == "__main__":
#     llama_embeddings = Llama_Embeddings("llama_7B","imdb")