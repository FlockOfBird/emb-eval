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
    def __init__(self):
        # logging configuration for better code monitoring
        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info('loading dataset')
        self.train_data, self.test_data = get_drugdiscription()

        logging.info('loading model and tokenizer')
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # self.test_embeddings()

        embeddings_train, embeddings_test = self.get_embeddings(save_embeddings=True)

    def get_embeddings(self, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            save_embeddings: T/F value to save the model in results directory
        '''
        embed_train_data = True
        embed_test_data = False

        if(embed_train_data):
            embeddings_train = []
            for data_row in tqdm(self.train_data):
                tokens = self.tokenizer.encode_plus(
                    data_row['Discription'],
                    add_special_tokens=True,
                    max_length=50,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = tokens['input_ids']
                token_type_ids = tokens['token_type_ids']

                # Obtain sentence embedding
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=token_type_ids)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embedding = embedding.squeeze()
                    embeddings_train.append(embedding)
        
        if(embed_test_data):
            embeddings_test = []
            for data_row in tqdm(self.test_data):
                tokens = self.tokenizer.encode_plus(
                    data_row['Discription'],
                    add_special_tokens=True,
                    max_length=50,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = tokens['input_ids']
                token_type_ids = tokens['token_type_ids']

                # Obtain sentence embedding
                with torch.no_grad():
                    outputs = self.model(input_ids, token_type_ids=token_type_ids)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1)
                    embedding = embedding.squeeze()
                    embeddings_test.append(embedding)

        if save_embeddings:
            embeddings_train = np.array(embeddings_train)
            train_data = pd.concat(
                [pd.DataFrame(self.train_data), pd.DataFrame(embeddings_train)], axis=1)
            # Save the scores to a CSV file
            print(train_data.head())
            train_data.to_csv(
                'results/embeddings/bert50mt_base_Discription_embeddings.csv', sep='\t')

        return np.array(embeddings_train), np.array(embeddings_test)


if __name__ == "__main__":
    bert_embeddings = Bert_Embeddings()
