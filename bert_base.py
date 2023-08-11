import logging
import json
from tqdm import tqdm

import torch
from transformers import BertModel, AutoTokenizer

import tensorflow as tf
import numpy as np
import pandas as pd

from data import *


class Bert_Embeddings:
    def __init__(self, model_name, dataset, dataset_name):
        # logging configuration for better code monitoring
        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info('loading dataset')
        self.dataset_name = dataset_name
        self.train_data, self.test_data = dataset["train"], dataset["test"]

        logging.info('loading model and tokenizer')
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model_name = model_name
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # self.test_embeddings()

        embeddings_train, embeddings_test = self.get_embeddings(save_embeddings=True)

    def get_embeddings(self, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            save_embeddings: T/F value to save the model in results directory
        '''
        embed_train_data = True
        embed_test_data = True

        if(embed_train_data):
            embeddings_train = []
            for data_row in tqdm(self.train_data):
                tokens = self.tokenizer.encode_plus(
                    data_row['text'],
                    add_special_tokens=True,
                    max_length=self.tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = tokens['input_ids'].to("cuda")
                token_type_ids = tokens['token_type_ids'].to("cuda")

                # Obtain sentence embedding
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=token_type_ids)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embedding = embedding.squeeze().to("cpu")
                    embeddings_train.append(np.array(embedding))
                

        
        if(embed_test_data):
            embeddings_test = []
            for data_row in tqdm(self.test_data):
                tokens = self.tokenizer.encode_plus(
                    data_row['text'],
                    add_special_tokens=True,
                    max_length=self.tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = tokens['input_ids'].to("cuda")
                token_type_ids = tokens['token_type_ids'].to("cuda")

                # Obtain sentence embedding
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=token_type_ids)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embedding = embedding.squeeze().to("cpu")
                    embeddings_test.append(np.array(embedding))


        if save_embeddings:
            logging.info('saving train data embeddings')
            print(len(embeddings_train))
            print(pd.DataFrame(embeddings_train).head())
            train_data = pd.concat([pd.DataFrame(self.train_data), pd.DataFrame(embeddings_train)], axis=1)
            print(train_data.head())
            train_data.to_csv(f'results/embeddings/{self.model_name}_base_{self.dataset_name}_embeddings_train.csv', sep='\t')

            logging.info('saving test data embeddings')
            test_data = pd.concat([pd.DataFrame(self.test_data), pd.DataFrame(embeddings_test)], axis=1)
            test_data.to_csv(f'results/embeddings/{self.model_name}_{self.dataset_name}_embeddings_test.csv', sep='\t')

        return np.array(embeddings_train), np.array(embeddings_test)


# if __name__ == "__main__":
#     bert_embeddings = Bert_Embeddings("bert", "imdb")
