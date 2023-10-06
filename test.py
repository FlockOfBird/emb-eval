'''
Write your tests here...
'''

import pandas as pd
import os
import glob
file = '/home/foroogh/llama_embedding/results/eval_results.csv'
df = pd.read_csv(file, sep='\t')
#define how to aggregate various fields
agg_functions = {'employee': 'first', 'sales': 'sum', 'returns': 'sum'}

#create new DataFrame by combining rows with same id values
df_new = df.groupby(df['model_name'])

#view new DataFrame
print(df_new)

# f1_mlp, accuracy_mlp, error_rate_mlp = None, None, None
# if ("mlp" in classifiers):
#     # classifier = MLPClassifier(hidden_layer_sizes=(200, 100, 100), random_state=1, max_iter=200).fit(X_train, y_train)
#     classifier = MLPClassifier(hidden_layer_sizes=(
#         200, 300, 200, 200), random_state=1, max_iter=200).fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     f1_mlp, accuracy_mlp, error_rate_mlp = self.compute_scores(
#         y_test, y_pred)

# f1_lr, accuracy_lr, error_rate_lr = None, None, None
# if ("lr" in classifiers):
#     classifier = LinearRegression().fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     y_pred = np.round(y_pred).astype(int)
#     f1_lr, accuracy_lr, error_rate_lr = self.compute_scores(
#         y_test, y_pred)

# f1_rf, accuracy_rf, error_rate_rf = None, None, None
# if ("rf" in classifiers):
#     classifier = RandomForestClassifier(
#         max_depth=2, random_state=0).fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     f1_rf, accuracy_rf, error_rate_rf = self.compute_scores(
#         y_test, y_pred)
