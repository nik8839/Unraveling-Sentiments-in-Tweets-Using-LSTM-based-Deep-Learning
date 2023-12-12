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



#%%time
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.Tweet), maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.Tweet), maxlen=MAX_SEQUENCE_LENGTH)
print("Training X Shape:",x_train.shape)
print("Testing X Shape:",x_test.shape)



#Label Encoder
encoder = LabelEncoder()
encoder.fit(train_data.Label.to_list())

y_train = encoder.transform(train_data.Label.to_list())
y_test = encoder.transform(test_data.Label.to_list())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10


#%%time
documents = [_text.split() for _text in train_data.Tweet] 
import gensim.models.word2vec




w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)


w2v_model.build_vocab(documents)




words = list(w2v_model.wv.key_to_index.keys())
vocab_size = len(words)
print("Vocab size", vocab_size)


#%%time
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)



similar_words = w2v_model.wv.most_similar("heat")
print(similar_words)



print("Tokenizer Vocabulary Size:", len(tokenizer.word_index))
print("Word2Vec Model Vocabulary Size:", len(w2v_model.wv))




# Get the common vocabulary between tokenizer and Word2Vec model
common_vocab = set(tokenizer.word_index.keys()) & set(w2v_model.wv.key_to_index.keys())

# Create a mapping from words to indices in the common vocabulary
word_to_index = {word: i + 1 for i, word in enumerate(common_vocab)}  # Add 1 to start index from 1

# Update tokenizer's word_index using the common vocabulary
tokenizer.word_index = word_to_index

# Create the embedding matrix
vocab_size = len(tokenizer.word_index) + 1  # Add 1 to account for the 0 index
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))

for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]




embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)



model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

# %%time
history = model.fit(x_train, y_train,
                    batch_size=1024,
                    epochs=15,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

# %%time
score = model.evaluate(x_test, y_test, batch_size=1024)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEGATIVE"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at} 

predict("healthy discussion")

model.save("history.h5")

# %%time
y_pred_1d = []
y_test_1d = list(test_data.Label)
scores = model.predict(x_test)
y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',cmap=plt.cm.Blues):
  
   

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(12,12))
plot_confusion_matrix(cnf_matrix, classes=train_data.Label.unique(), title="Confusion matrix")
plt.show()

print(classification_report(y_test_1d, y_pred_1d))

accuracy_score(y_test_1d, y_pred_1d)





