import logging
import json
import argparse

import torch
from transformers import BertModel, AutoTokenizer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC

from imdb_data import get_imdb, get_small_imdb

# logging configuration for better code monitoring
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--Plot", help = "Draw plot for embeddings")
args = parser.parse_args()
 
# Set device to utilize GPU
device = "auto"

logging.info('loading dataset')
train_data, test_data = get_imdb()

logging.info('loading model and tokenizer')
model = BertModel.from_pretrained("bert-base-uncased") # The bare Bert Model transformer outputting raw hidden-states without any specific head on top.
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

logging.info('encoding data and generating embeddings')
review_embeddings_train = []
for i, review in enumerate(train_data):
    tokens = bert_tokenizer.encode_plus(
        review['text'], 
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
        outputs = model(input_ids, token_type_ids=token_type_ids)
        review_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        review_embedding = review_embedding.squeeze()
        review_embeddings_train.append(review_embedding)

review_embeddings_test = []
for i, review in enumerate(test_data):
    tokens = bert_tokenizer.encode_plus(
        review['text'], 
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
        outputs = model(input_ids, token_type_ids=token_type_ids)
        review_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        review_embedding = review_embedding.squeeze()
        review_embeddings_test.append(review_embedding)

if args.Plot:
    logging.info('creating plot')
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)
    review_embeddings_pca = pca.fit_transform(review_embeddings_test)

    # Plot the reduced-dimensional embeddings
    for i, review in enumerate(test_data):
        color = 'red' if review['label'] else 'blue'
        plt.scatter(review_embeddings_pca[i, 0], review_embeddings_pca[i, 1], color=color)

    plt.title("Sentence Embeddings (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig('results/bert_base_pca.png')

    # Perform dimensionality reduction using TSNE
    tsne = TSNE(n_components=2)
    review_embeddings_tsne = torch.stack(review_embeddings_test)
    review_embeddings_tsne = tsne.fit_transform(review_embeddings_tsne)

    # Plot the reduced-dimensional embeddings
    for i, review in enumerate(test_data):
        color = 'red' if review['label'] else 'blue'
        plt.scatter(review_embeddings_tsne[i, 0], review_embeddings_tsne[i, 1], color=color)

    plt.title("Sentence Embeddings (TSNE)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig('results/bert_base_tsne.png')

logging.info('classification using SVM')
svm = SVC()

train_labels = []
for review in train_data:
    train_labels.append(review['label'])

test_labels = []
for review in test_data:
    test_labels.append(review['label'])

svm.fit(review_embeddings_train, train_labels)

# Make predictions on the test data
y_pred = svm.predict(review_embeddings_test)

# Calculate F1 and accuracy score
f1 = f1_score(test_labels, y_pred)
accuracy = accuracy_score(test_labels, y_pred)

# experiment descriptions
scores = {
    'Data': 'IMDB',
    'Data Size Test': len(review_embeddings_test),
    'Data Size Train': len(review_embeddings_train),
    'F1 score': f1,
    'Accuracy': accuracy
}

# Save the scores to a JSON file
with open('results/bert_base_results.json', 'w') as file:
    json.dump(scores, file)

print(scores)