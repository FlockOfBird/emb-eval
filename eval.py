from typing import List, Dict
import re
import os
import glob
from tqdm import tqdm

import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Evlaution:
    def __init__(self):

        df_results = pd.DataFrame(columns=['model_name', 'dataset',
                                           'f1_mlp', 'accuracy_mlp', 'error_rate_mlp',
                                           'f1_lr', 'accuracy_lr', 'error_rate_lr',
                                           'f1_rf', 'accuracy_rf', 'error_rate_rf',
                                           'f1_svm', 'accuracy_svm', 'error_rate_svm',
                                           'test#', 'train#'])

        classifiers = ["rf", "lr", "mlp", "svm"] #
        datasets = ["yelpf", "agnews", "yelpp", "imdb"] #
        models = ["llama-7B", "llama2-7B", "bert"] #
        method = "k_fold" # "k_fold", "train_test"

        for classifier in classifiers:
            for dataset in datasets:
                for model in models:

                    df_train, df_test = self.load_data(dataset, model)

                    train_labels = []
                    for i, data_row in df_train.iterrows():
                        train_labels.append(data_row['label'])

                    test_labels = []
                    for i, data_row in df_test.iterrows():
                        test_labels.append(data_row['label'])

                    # getting only train embeddings
                    df_train = df_train.iloc[:, 2:]
                    # getting only test embeddings
                    df_test = df_test.iloc[:, 2:]

                    X = pd.concat([df_train, df_test])
                    del df_train, df_test

                    y = train_labels + test_labels
                    y = np.array(y)

                    results = []

                    if (method == "k_fold"):
                        k = 5
                        kf = KFold(n_splits=k, random_state=None)
                        i = 0
                        for train_index, test_index in tqdm(kf.split(X)):
                            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                            y_train, y_test = y[train_index], y[test_index]
                            i = i + 1
                            print(f'evaluating {dataset} with {model} with {classifier} using {method} at {i}th fold of {k}')
                            result = self.eval(X_train, X_test, y_train, y_test, model, dataset, classifier)
                            results.append(result)
                            df_results.loc[len(df_results)] = result
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
                        print(f'evaluating {dataset} with {model} with {classifier} using {method}')
                        result = self.eval(X_train, X_test, y_train, y_test, model, dataset, classifier)
                        results.append(result)
                        df_results.loc[len(df_results)] = result
                        # print(df_results)

        # print(df_results)
        path = "./results/eval_plots/eval_results_test.csv"
        df_results.to_csv(path, index=False, sep='\t')
        self.plot_results(path, method)

    def eval(self, X_train, X_test, y_train, y_test, model, dataset, classifier, draw_cm=False, save_scores=False) -> Dict:
        """
            draw_cm:        draw confusin matrix
            classifiers:    List of classifiers (mlp, lr, rf)
        """

        emb_cls = {
            "mlp":  MLPClassifier(hidden_layer_sizes=(200, 100, 100), max_iter=200),
            "lr":   LogisticRegression(),
            "rf":   RandomForestClassifier(max_depth=5),
            "svm":  make_pipeline(StandardScaler(), SVC(gamma='auto'))
        }

        emb_cls = emb_cls.get(classifier).fit(X_train, y_train)
        y_pred = emb_cls.predict(X_test)
        y_pred = np.round(y_pred).astype(int) if classifier == "lr" else y_pred
        f1, accuracy, error_rate = self.compute_scores(y_test, y_pred)

        if (draw_cm):
            print('plotting confusion matrix')
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
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
        test_path, train_path = None, None
        print(dataset_name, model)
        for path in glob.iglob(f'{BASE_PATH}/*'):
            if (dataset_name in path and model in path and 'test' in path):
                test_path = path
            if (dataset_name in path and model in path and 'train' in path):
                train_path = path

        df_train = pd.read_csv(train_path, sep='\t')
        df_test = pd.read_csv(test_path, sep='\t')

        return df_train, df_test

    def plot_results(self, path, method):
        df = pd.read_csv(path, sep='\t')

        # merge rows
        if("k_fold" not in method):
            df.fillna(0, inplace=True)
            df = df.groupby(['model_name', 'dataset']).sum().reset_index()

        models = ["bert", "llama-7B", "llama2-7B"]
        datasets = ["imdb", "yelpp", "yelpf", "agnews"]
        metric_clss = [
                        "f1_mlp", "accuracy_mlp", "error_rate_mlp",
                        "f1_lr", "accuracy_lr", "error_rate_lr",
                        'f1_svm', 'accuracy_svm', 'error_rate_svm',
                        "f1_rf", "accuracy_rf", "error_rate_rf"
                    ]
        model_ranks = {
            'bert': 0,
            'llama-7B': 0,
            'llama2-7B': 0
        }
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
                df_cleaned = df[df['dataset'] == dataset].sort_values(
                    by=['model_name'])[['model_name', metric_cls]]
                df_cleaned = df_cleaned.reset_index()

                if("k_fold" in method):
                    df_cleaned = df_cleaned.drop(['index'], axis=1)
                    df_cleaned = df_cleaned.dropna()

                    for model in models:
                        mean = df_cleaned[df_cleaned['model_name'] == model][metric_cls].mean()
                        df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned['model_name'] == model].index)
                        new_row = {'model_name': model, metric_cls: mean}
                        df_cleaned.loc[len(df_cleaned)] = new_row

                    df_cleaned = df_cleaned.round(3)
                    # print(df_results_db)
                    b = np.squeeze(df_cleaned.iloc[:, 1:].values)
                else:
                    b = np.squeeze(df_cleaned.iloc[:, 2:].values)
                # find which model did best
                print('b:', b, dataset)
                best_model = np.argmax(b)
                invert_index = {
                    0: 'bert',
                    1: 'llama-7B',
                    2: 'llama2-7B'
                }
                best_model = invert_index[best_model]
                if(metric_cls == 'accuracy_mlp' or dataset == 'imdb'):
                    model_ranks[best_model] = model_ranks[best_model] + 1
                models_results['bert'].append(b[0])
                models_results['llama-7B'].append(b[1])
                models_results['llama2-7B'].append(b[2])
                print(model_ranks)

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
            # plt.savefig(f'results/eval_plots/{metric_cls}_test.png')

    def plots_max(self, path, method):
        df = pd.read_csv(path, sep='\t')
        print(df)

if __name__ == "__main__":
    eval = Evlaution()
