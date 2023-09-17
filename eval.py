from typing import List, Dict
import re
import os
import glob
from tqdm import tqdm

import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evlaution:
    def __init__(self):

        df_results = pd.DataFrame(columns=['model_name', 'dataset', 'f1_mlp', 'accuracy_mlp', 'error_rate_mlp',
                                  'f1_lr', 'accuracy_lr', 'error_rate_lr', 'f1_rf', 'accuracy_rf', 'error_rate_rf', 'test#', 'train#'])

        classifiers = ["lr", "rf", "mlp"]
        datasets = ["agnews", "yelpf", "yelpp", "imdb"]
        models = ["llama-7B", "llama-13B", "llama-30B", "llama-65B",
                  "llama2-7B", "llama2-13B", "llama2-70B", "bert"]

        for classifier in classifiers:
            for dataset in datasets:
                for model in models:

                    df_train, df_test, model, dataset = self.load_data(
                        dataset, model)

                    print(
                        f'evaluating {dataset} with {model} with {classifier}')

                    k = 10
                    kf = KFold(n_splits=k, random_state=None)

                    train_labels = []
                    for i, data_row in df_train.iterrows():
                        train_labels.append(data_row['label'])

                    test_labels = []
                    for i, data_row in df_test.iterrows():
                        test_labels.append(data_row['label'])

                    # getting only train embeddings
                    df_train = df_train.iloc[:, 3:]
                    # getting only test embeddings
                    df_test = df_test.iloc[:, 3:]
                    df = pd.concat([df_train, df_test])

                    y = train_labels + test_labels
                    y = np.array(y)

                    results = []
                    for train_index, test_index in tqdm(kf.split(df)):
                        X_train, X_test = df.iloc[train_index,
                                                  :], df.iloc[test_index, :]
                        y_train, y_test = y[train_index], y[test_index]
                        result = self.eval(
                            X_train, X_test, y_train, y_test, model, dataset, classifier)
                        results.append(result)
                        df_results.loc[len(df_results)] = result

        print(df_results)
        # df_results.to_csv(
        #     "./results/eval_results_10fold_mlp.csv", index=False, sep='\t')

    def eval(self, X_train, X_test, y_train, y_test, model, dataset, classifier, draw_cm=False, save_scores=False) -> Dict:
        """
            draw_cm:        draw confusin matrix
            classifiers:    List of classifiers (mlp, lr, rf)
        """

        emb_cls = {
            "mlp":  MLPClassifier(hidden_layer_sizes=(200, 100, 100), max_iter=200),
            "lr":   LinearRegression(),
            "rf":   RandomForestClassifier(max_depth=2)
        }

        emb_cls = emb_cls.get(classifier).fit(X_train, y_train)
        print(emb_cls)
        y_pred = emb_cls.predict(X_test)
        y_pred = np.round(y_pred).astype(int) if classifier == "lr" else y_pred
        f1, accuracy, error_rate = self.compute_scores(y_test, y_pred)

        if (draw_cm):
            print('plotting confusion matrix')
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=np.unique(y_test))
            fig, ax = plt.subplots(figsize=(3, 3))
            disp.plot(ax=ax)

        stats = {
            'model_name':               model,
            'dataset':                  dataset,
            'classifier':               classifier,

            f'f1_'+classifier:          f1,
            f'accuracy_'+classifier:    accuracy,
            f'error_rate_'+classifier:  error_rate,

            'test#':                    len(X_test),
            'train#':                   len(X_train)
        }
        print(stats)
        return stats

    def compute_scores(self, Y, y):
        # Calculate F1 and accuracy score
        f1 = f1_score(Y, y, average='micro')
        accuracy = accuracy_score(Y, y)
        # Calculate error rate
        cm = confusion_matrix(Y, y, labels=np.unique(Y))
        # print(cm)
        total_misclassified = sum(cm[i][j] for i in range(
            len(cm)) for j in range(len(cm)) if i != j)
        total_instances = sum(sum(row) for row in cm)
        # print(total_misclassified,total_instances)
        er = total_misclassified / total_instances

        return f1, accuracy, er

    def load_data(self, dataset_name, model):
        BASE_PATH = './results/embeddings/'
        print('1')
        for TEST_PATH in glob.iglob(f'{BASE_PATH}/*_test_embeddings.csv'):
            print('2')
            print(TEST_PATH)
            print(model)
            print(model in TEST_PATH)
            print(dataset_name)
            print(dataset_name in TEST_PATH)

            # if (dataset_name in TEST_PATH and model in TEST_PATH):
            #     dataset_patern = TEST_PATH.replace('_test_embeddings.csv', '')
            #     test_p = TEST_PATH
            #     print('fs')
            #     for TRAIN_PATH in glob.iglob(f'{dataset_patern}_train_embeddings.csv'):
            #         print('Fu', TRAIN_PATH)
            #         if (dataset_name in TRAIN_PATH and model in TRAIN_PATH):
            #             train_p = TRAIN_PATH
            #             break

        print(train_p, test_p)
        df_train = pd.read_csv(train_p, sep='\t')
        df_train = df_train.drop(['Unnamed: 0.1'], axis=1)
        df_test = pd.read_csv(test_p, sep='\t')
        df_test = df_test.drop(['Unnamed: 0.1'], axis=1)

        # combine both train and test dataset and create random splits from these datasets
        # ...

        # take dataset name out of path
        splited_path = TRAIN_PATH.split("_")
        model = splited_path[0].replace(BASE_PATH, '')
        dataset = splited_path[1]

        return df_train, df_test, model, dataset

    def plot_results(df_results):
        models = ["bert", "llama-7B", "llama2-7B"]
        datasets = ["imdb", "yelpp", "yelpf", "agnews"]
        metric_clss = ["f1_mlp", "accuracy_mlp", "error_rate_mlp", "f1_lr",
                       "accuracy_lr", "error_rate_lr", "f1_rf", "accuracy_rf", "error_rate_rf"]

        for metric_cls in metric_clss:
            x = np.arange(len(datasets))  # the label locations
            width = 0.32  # the width of the bars
            multiplier = 0
            fig, ax = plt.subplots(layout='constrained')
            models_results = {
                'bert': [],
                'llama-7B': [],
                'llama2-7B': []
            }

            for dataset in datasets:
                df_results_db = df_results[df_results['dataset'] == dataset].sort_values(
                    by=['model_name'])[['model_name', metric_cls]]
                df_results_db = df_results_db.reset_index()
                b = np.squeeze(df_results_db.iloc[:, 2:].values)
                models_results['bert'].append(b[0])
                models_results['llama-7B'].append(b[1])
                models_results['llama2-7B'].append(b[2])

            for attribute, value in models_results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, value, width, label=attribute)
                ax.bar_label(rects, padding=3)
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(metric_cls)
            ax.set_title(f'Model performance on each dataset by {metric_cls}')
            ax.set_xticks(x + width, datasets)
            ax.legend(loc='upper right', ncols=3)
            ax.set_ylim(0, 1)

            # plt.show()
            plt.savefig(f'results/eval_plots/{metric_cls}.png')


if __name__ == "__main__":
    eval = Evlaution()
