import logging
import json
import argparse
from tqdm import tqdm 

import torch
from transformers import BertModel, AutoTokenizer

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC

from imdb_data import get_imdb, get_small_imdb

class Bert_Embeddings:
    def __init__(self):
        # logging configuration for better code monitoring
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        # Read arguments from command line
        parser = argparse.ArgumentParser()
        parser.add_argument("--Plot", help = "Draw plot for embeddings")
        args = parser.parse_args()

        logging.info('loading dataset')
        self.train_data, self.test_data = get_small_imdb(50)

        logging.info('loading model and tokenizer')
        self.model = BertModel.from_pretrained("bert-base-uncased") # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        embeddings_train, embeddings_test = self.get_embeddings(
            True, False, save_embeddings=True)

        if args.Plot:
            # self.plot_embeddings(embeddings_test)
            self.plot_embeddings(embeddings_train)

    def get_embeddings(self, train, test, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            train: T/F value checks if we should generate embeddings for train data
            test: T/F value checks if we should generate embeddings for train data
            save_embeddings: T/F value to save the model in results directory
        '''
        embeddings_train = []
        for data_row in tqdm(self.train_data):
            tokens = self.tokenizer.encode_plus(
                data_row['text'], 
                add_special_tokens=True, 
                max_length=512, 
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

        embeddings_test = []
        for data_row in tqdm(self.test_data):
            tokens = self.tokenizer.encode_plus(
                data_row['text'], 
                add_special_tokens=True, 
                max_length=512, 
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
            train_data = pd.concat([pd.DataFrame(self.train_data), pd.DataFrame(embeddings_train)], axis=1)
            # Save the scores to a CSV file
            print(train_data.head())
            train_data.to_csv('results/embeddings/test_bert_base_embeddings.csv', sep='\t')

        return np.array(embeddings_train), np.array(embeddings_test)

    def plot_embeddings(self, embeddings_test):
        logging.info('creating plot')
        # Perform dimensionality reduction using PCA
        pca = PCA(n_components=2)
        
        embeddings_pca = pca.fit_transform(embeddings_test)
        print(embeddings_pca.shape)


        # Plot the reduced-dimensional embeddings
        for i, data_row in enumerate(self.test_data):
            color = 'red' if data_row['label'] else 'blue'
            plt.scatter(embeddings_pca[i, 0], embeddings_pca[i, 1], color=color)

        plt.title("Sentence Embeddings (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig('results/test_bert_base_pca.png')

        # Perform dimensionality reduction using TSNE
        tsne = TSNE(n_components=2)
        
        embeddings_tsne = torch.stack(embeddings_test)
        embeddings_tsne = tsne.fit_transform(embeddings_tsne)

        # Plot the reduced-dimensional embeddings
        for i, data_row in enumerate(self.test_data):
            color = 'red' if data_row['label'] else 'blue'
            plt.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], color=color)

        plt.title("Sentence Embeddings (TSNE)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig('results/test_bert_base_tsne.png')

    def evaluate_embeddings(self, embeddings_train, embeddings_test):
        logging.info('classification using SVM')
        svm = SVC()
        
        train_labels = []
        for data_row in self.train_data:
            train_labels.append(data_row['label'])

        test_labels = []
        for data_row in self.test_data:
            test_labels.append(data_row['label'])

        svm.fit(embeddings_train, train_labels)

        # Make predictions on the test data
        y_pred = svm.predict(embeddings_test)

        # Calculate F1 and accuracy score
        f1 = f1_score(test_labels, y_pred)
        accuracy = accuracy_score(test_labels, y_pred)

        # experiment descriptions
        scores = {
            'Data': 'IMDB',
            'Data Size Test': len(embeddings_test),
            'Data Size Train': len(embeddings_train),
            'F1 score': f1,
            'Accuracy': accuracy
        }

        # Save the scores to a JSON file
        with open('results/test_bert_base_results.json', 'w') as file:
            json.dump(scores, file)

        print(scores)

if __name__ == "__main__":
    bert_embeddings = Bert_Embeddings()