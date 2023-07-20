import logging
import json
import argparse
from tqdm import tqdm

import torch
from transformers import BertModel, AutoTokenizer

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from data import *


class Bert_Embeddings:
    def __init__(self):
        # logging configuration for better code monitoring
        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)

        # Read arguments from command line
        parser = argparse.ArgumentParser()
        parser.add_argument("--Plot", help="Draw plot for embeddings")
        args = parser.parse_args()

        logging.info('loading dataset')
        self.train_data, self.test_data = get_sample_data()

        logging.info('loading model and tokenizer')
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # self.test_embeddings()

        embeddings_train, embeddings_test = self.get_embeddings(
            save_embeddings=True)

        if args.Plot:
            # self.plot_embeddings(embeddings_test, self.test_data)
            self.plot_embeddings(embeddings_train, self.train_data)

    def get_embeddings(self, save_embeddings):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            save_embeddings: T/F value to save the model in results directory
        '''
        embeddings_train = []
        for data_row in tqdm(self.train_data):
            tokens = self.tokenizer.encode_plus(
                data_row['text'],
                add_special_tokens=True,
                max_length=64,
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
                max_length=64,
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
                'results/embeddings/test_bert_base_embeddings.csv', sep='\t')

        return np.array(embeddings_train), np.array(embeddings_test)

    def plot_embeddings(self, embeddings, labeled_data):
        logging.info('creating plot')
        # Perform dimensionality reduction using PCA
        pca = PCA(n_components=2)

        embeddings_pca = pca.fit_transform(embeddings)

        # Plot the reduced-dimensional embeddings
        for i, data_row in enumerate(labeled_data):
            color = 'red' if data_row['label'] else 'blue'
            plt.scatter(embeddings_pca[i, 0],
                        embeddings_pca[i, 1], color=color, label=data_row['text'])
            plt.annotate(text=data_row['text'],
                         xy=(embeddings_pca[i, 0], embeddings_pca[i, 1]))

        # plt.legend()
        plt.title("Sentence Embeddings (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig('results/test_bert_base_pca.png')

        # Perform dimensionality reduction using TSNE
        tsne = TSNE(n_components=2)
        embeddings_tsne = tf.convert_to_tensor(embeddings)
        embeddings_tsne = tsne.fit_transform(embeddings_tsne)

        # Plot the reduced-dimensional embeddings
        for i, data_row in enumerate(labeled_data):
            color = 'red' if data_row['label'] else 'blue'
            plt.scatter(embeddings_tsne[i, 0],
                        embeddings_tsne[i, 1], color=color)
            plt.annotate(text=data_row['text'],
                         xy=(embeddings_tsne[i, 0], embeddings_tsne[i, 1]))

        # plt.legend()
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

    def test_embeddings(self):
        def get_embeddings(input):
            tokens = self.tokenizer.encode_plus(
                input,
                add_special_tokens=True,
                max_length=10,  # very important for the result of cosine similarity
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids']
            token_type_ids = tokens['token_type_ids']
            attention_mask = tokens['attention_mask']
            # Obtain sentence embedding
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=token_type_ids)
                embedding = torch.mean(outputs.last_hidden_state, dim=1)
                embedding = embedding.squeeze()
            print(embedding.shape)
            return embedding

        train_data, test_data = get_sample_data()

        embeddings = []
        sentences = []
        for data_row in train_data:
            embedding = get_embeddings(data_row['text'])
            sentences.append(data_row['text'])
            embeddings.append(embedding)

        def get_cosine_similarity(feature_vec_1, feature_vec_2):
            output = cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))
            return output[0]

        cs_values = []
        for i in embeddings:
            cosine_similarity_values = []
            for j in embeddings:
                cosine_similarity_values.append(get_cosine_similarity(i, j))
            cs_values.append(cosine_similarity_values)

        cs_values = np.array(cs_values)
    
        print('final matrix:', cs_values.shape)
        sentences = np.array(sentences)

        plt.figure(figsize = (14,14))
        plt.imshow(cs_values, cmap='Blues_r', interpolation='nearest')
        # Add color bar to indicate the similarity scale
        plt.colorbar()

        # Set the tick labels for x and y axes
        plt.xticks(np.arange(len(sentences)), sentences, rotation=45)
        plt.yticks(np.arange(len(sentences)), sentences)

        # Set the plot title and labels
        plt.title('Cosine Similarity Heatmap')
        plt.xlabel('Sentences')
        plt.ylabel('Sentences')

        plt.savefig('results/bert_base_cosine_similarity.png')

if __name__ == "__main__":
    bert_embeddings = Bert_Embeddings()
