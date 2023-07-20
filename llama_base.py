import logging
import json
import argparse
from tqdm import tqdm

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

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

class Llama_Embeddings:
    def __init__(self):
        # logging configuration for better code monitoring
        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)

        # Read arguments from command line
        parser = argparse.ArgumentParser()
        parser.add_argument("--Plot", help="Draw plot for embeddings")
        args = parser.parse_args()

        # Set device to auto to utilize GPU
        device = "auto"

        PATH_TO_CONVERTED_WEIGHTS = "./7B_converted/"

        logging.info('loading dataset')
        self.train_data, self.test_data = get_sample_data()

        # Use GPU runtime an High_RAM before run this pieace of code
        # Default CUDA device
        print('if cuda is available:', torch.cuda.is_available())
        # returns 0 in my case
        print('current cuda device:', torch.cuda.current_device())
        # returns 1 in my case
        print('number of cuda devices', torch.cuda.device_count())

        logging.info('loading model and tokenizer')
        self.model = LlamaForCausalLM.from_pretrained(
            PATH_TO_CONVERTED_WEIGHTS,
            device_map=device
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            PATH_TO_CONVERTED_WEIGHTS
        )

        # self.test_embeddings()

        embeddings_train, embeddings_test = self.get_embeddings(save_embeddings=False)

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
            # print(data_row)
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
            embeddings_train = np.array(embeddings_train)
            train_data = pd.concat([pd.DataFrame(self.train_data), pd.DataFrame(embeddings_train)], axis=1)
            # Save the scores to a CSV file
            print(train_data.head())
            train_data.to_csv('results/embeddings/llama_base_Discription_embeddings.csv', sep='\t')

        return embeddings_train, embeddings_test

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
        plt.savefig('results/test_llama_base_pca.png')

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
        plt.savefig('results/test_llama_base_tsne.png')

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
        with open('results/test_llama_base_output_embeddings_results.json', 'w') as file:
            json.dump(scores, file)

        print(scores)

    def test_embeddings(self):
        # prompt = "If we want to compare Coldplay with Arctict Monkeys"
        # inputs = self.tokenizer(prompt, return_tensors="pt")

        # simple code to generate code and check the model funtionality
        # sending inputs to gpu to utilize the gpu speed
        # input_ids = inputs.input_ids.to('cuda')
        # generate_ids = self.model.generate(input_ids, max_length=30).to("cuda")
        # decoded_text = self.tokenizer.batch_decode(
        #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(decoded_text)

        def get_embeddings(prompt):
            input_ids = self.tokenizer(prompt, max_length=10).input_ids
            # we can use get_output_embeddings as well
            input_embeddings = self.model.get_input_embeddings()
            embeddings = input_embeddings(torch.LongTensor([input_ids]))
            mean = torch.mean(embeddings[0], 0).cpu().detach()
            print(mean.shape)
            # print(embeddings.shape)
            return mean
        
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
        print('final matrix:', cs_values)

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

        plt.savefig('results/llama_base_cosine_similarity.png')


if __name__ == "__main__":
    llama_embeddings = Llama_Embeddings()
