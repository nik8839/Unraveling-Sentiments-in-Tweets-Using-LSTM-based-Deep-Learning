# DataFrame
import pandas as pd

# Matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

import sklearn
import matplotlib
import numpy
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools


# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')

#Versions 
print(sklearn.__version__)
print(matplotlib.__version__)
print(numpy.__version__)
print(pd.__version__)
print(nltk.__version__)

df = pd.read_csv("/kaggle/input/tweetanalysis/dataset.csv", encoding='latin-1', header=None) #read csv file without header as dataframe
from sklearn.feature_extraction.text import TfidfVectorizer #  import TF-idf vectorizer
from sklearn.utils import shuffle
df = shuffle(df)

print("Dataset size:", len(df))
df.head()

df.columns = ["Label", "people_id", "Date", "query", "user", "Tweet"] # give column names
#data
df.head()

df = df.drop(['people_id', 'Date', 'query', 'user'], axis=1)
df.head()

print(df.columns)



target_cnt = Counter(df.Label)

plt.figure(figsize=(16,8))
plt.ylabel('Counts')
plt.xlabel('Labels')
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")

df.describe()

decode_map = {0: "NEGATIVE", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
df.Label = df.Label.apply(lambda x: decode_sentiment(x))
df.head()

df.info() #printing information

df['Label'].unique() #unique labels

df.head()

plt.boxplot(df.Length_of_Tweet) # plotting pre_clean_len column
plt.show()

import random
random_idx_list = [random.randint(1,len(df.Label)) for i in range(10)] # creates random indexes to choose from dataframe
df.loc[random_idx_list,:].head(10) # Returns the rows with the index and display it

df[df.Length_of_Tweet > 350].head(10)

stop_words = stopwords.words('english') #list of stopwords
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words: #discarding stopwords
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

df.Tweet = df.Tweet.apply(lambda x: preprocess(x))

df.head()

from wordcloud import WordCloud

plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Label == 'POSITIVE'].Tweet))
plt.imshow(wc , interpolation = 'bilinear')


plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Label == 'NEGATIVE'].Tweet))
plt.imshow(wc , interpolation = 'bilinear')

TRAIN_SIZE = 0.85
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 50

train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE,
                                         random_state=1) # Splits Dataset into Training and Testing set
print("Train Data size:", len(train_data))
print("Test Data size", len(test_data))

train_data.head(20)

test_data.head(20)

#%%time
from keras.preprocessing.text import Tokenizer #preprocessing text using tokenization

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.Tweet)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)