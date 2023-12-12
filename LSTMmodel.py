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