from data import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_word_range(num_words_series, max_num_words):
    """
    A function to plot a histogram to show the frequency of occurence of word-counts in a text series.
        
    Arguments:
        num_words_series : pd.Series | text Series to count the number of words in each of its instances. 
        
    Returns:
        Histogram plot of the word_counts for each instance in the text Series.         
    """
    plt.figure(figsize=(15,7))
    bins = np.linspace(0, max_num_words , 20)

    plt.hist(num_words_series, bins=bins, alpha=0.5, histtype='bar', ec='black')
    plt.xticks(ticks=np.linspace(0,max_num_words,num=20),rotation=90)
    plt.title(f'Word-Count for each review')
    plt.xlabel('Number of words')
    plt.ylabel('Frequency')

    plt.savefig(f'./data_investigation/{dataset_name}_num_of_words.png')

# getting the instance number of words:
datasets = ["yelpf", "yelpp", "imdb", "agnews", "sst2"]
average_len = {}
max_len = {}
for dataset_name in datasets:
    data = get_dataset(dataset_name)
    print(dataset_name)
    print('# of train:', len(data["train"]))
    print('# of test:', len(data["test"]))
    print('total # of instances:', len(data["train"]) + len(data["test"]))

    train_data = pd.DataFrame(data["train"])

    num_words_series = train_data['text'].apply( lambda x: len(x.split()) )
    average_len[dataset_name] = int(num_words_series.mean())
    max_len[dataset_name] = int(num_words_series.max())

    # plot_word_range(num_words_series, max_num_words)

datasets = list(average_len.keys())
values = list(average_len.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(datasets, values, width = 0.4)
 
plt.xlabel("Datasets")
plt.ylabel("average length")
plt.title("Average length of words for each dataset")
plt.savefig(f'./data_investigation/average_of_words.png')

datasets = list(max_len.keys())
values = list(max_len.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(datasets, values, width = 0.4)
 
plt.xlabel("Datasets")
plt.ylabel("max length")
plt.title("Max length of words for each dataset")
plt.savefig(f'./data_investigation/max_of_words.png')