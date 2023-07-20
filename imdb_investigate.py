from data import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# getting the instance number of words:
train_data, test_data = get_imdb()
train_data = pd.DataFrame(train_data)

def plot_word_range(num_words_series):
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
    plt.xticks(ticks=np.linspace(0,2500,num=20),rotation=90)
    plt.title(f'Word-Count for each review')
    plt.xlabel('Number of words')
    plt.ylabel('Frequency')

    return plt.savefig('imdb_dataset/imdb_num_of_words.png')


num_words_series = train_data['text'].apply( lambda x: len(x.split()) )
max_num_words = num_words_series.max()
print(num_words_series, "\n \n" + "max number of words is: ", max_num_words)

plot_word_range(num_words_series)