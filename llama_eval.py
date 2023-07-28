import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

if args.TestEmbeddings:
    self.test_embeddings()
if args.Plot:
    self.plot_embeddings(embeddings_test, self.test_data)
    self.plot_embeddings(embeddings_train, self.train_data)
    
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
    with open('results/llama65b/llama_base_output_embeddings_results.json', 'w') as file:
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
        print(prompt)

        print('embs shape:', embeddings.shape)
        print('embs:',embeddings)
        # we use embeddings[0] since the input_embeddings could return multiple word embeddings at the same time, but since we are passing one input_ids at a time it returns only one sequence embedding in its zero index. 
        # second 0 in (embedding[0], 0) indicates the output dimension of mean calculation
        mean = torch.mean(embeddings[0], 0).cpu().detach() 
        print('mean shape',mean.shape)
        print('mean ',mean)
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