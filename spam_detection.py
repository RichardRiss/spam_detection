# General stuff
import logging
import os
import sys
import glob

# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')
 
# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
import warnings
warnings.filterwarnings('ignore')
import re
from email.parser import BytesParser
from email.policy import default




############################
# Import Data
############################

def importData():
  # Spam Ham data
  #spam_ham = pd.read_csv('./Spam_Ham_data.csv')
  #spam = spam_ham.loc[spam_ham.label == 1.0]
  #ham = spam_ham.loc[spam_ham.label == 0]

  # Other Spam data
  csv_files = glob.glob(os.path.join('./Data', "*.csv"))
  spam_data = []
  encodings = ["utf-8",'unicode_escape', "utf-8-sig", "latin1", "cp1252","iso-8859-1"]
  encoding_dict = {}
  for f in csv_files:
    for encoding in encodings:
      try:
        data = pd.read_csv(f,encoding=encoding, on_bad_lines='skip')
        spam_data.append(data)
        encoding_dict[f]=encoding
        break
      except Exception as e:  # or the error you receive
          pass
      
  # Enron Ham Data
  ham = pd.read_csv('./enron.csv')
  
  spam = pd.concat(spam_data)


  logging.info(f'{len(spam)} Spam sets and {len(ham)} valid sets.')
  data = pd.concat([ham[['email','label']], spam[['email','label']]], axis=0, ignore_index=True)
  return data



############################
# Data Preprocessing
############################

filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
stop_words = stopwords.words('english')
def clean_text(email):
    text = re.sub('[{}]'.format(filter),'',email)
    text = re.sub('\n+','',text)
    text = re.sub('\\s+',' ',text)
    text = re.sub(r'[0-9]+','number',text)
    text = text.lower()
    text = re.sub(r'[a-z0-9]{12,}','',text)
    text = re.sub('\\s+',' ',text)
    return text


def stopword(text):
    return ' '.join([w for w in text.split() if w not in stop_words])


def dataPreprocessing(data: pd.DataFrame):
  data['content'] = None
  for i in range(data.shape[0]):
    email = BytesParser(policy=default).parsebytes(data['email'][i].encode('utf-8'))
    payload = email.get_payload()
    data['content'][i] = stopword(clean_text(str(payload)))
  
  data = data.dropna(axis=0)
  data = data.reset_index(drop=True)
  return data




############################
# Train data
############################
def tokenize(clean_data):
    # Split into features and label
    X = clean_data.content
    y = clean_data.label.astype(int)
    
    #train test split
    train_X, test_X, train_label, test_label = train_test_split(X,
                                                        y,
                                                        test_size = 0.2,
                                                        random_state = 42)

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_X)
    
    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)
    
    # Pad sequences to have the same length
    max_len = 100  # maximum sequence length
    train_sequences = pad_sequences(train_sequences,
                                    maxlen=max_len,
                                    padding='post',
                                    truncating='post')
    test_sequences = pad_sequences(test_sequences,
                                maxlen=max_len,
                                padding='post',
                                truncating='post')

    
    return train_sequences, test_sequences, train_label, test_label, tokenizer


def train_model(train_sequences, test_sequences, train_label, test_label, tokenizer):
   # Build the model
    max_len = len(train_sequences[0])
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                        output_dim=32,
                                        input_length=max_len))
    model.add(tf.keras.layers.LSTM(16))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Print the model summary
    logging.info(model.summary())

    # compile model
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'],
              optimizer = 'adam')
   
    # Decrease learning rate per epoch when learning rate decreases
    lr = ReduceLROnPlateau(patience = 2, monitor = 'val_loss', factor = 0.5, verbose = 0)

    # Stop model if learning stops improving
    es = EarlyStopping(patience=3, monitor = 'val_accuracy', restore_best_weights = True)

    
    # Train the model
    history = model.fit(train_sequences, train_label, validation_data=(test_sequences, test_label),
                        epochs=20,
                        batch_size=32,
                        callbacks = [lr, es]
                    )
    
    logging.info(history)

    return model


############################
# Evaluate model
############################
def evaluate(model, test_sequences, test_label):
    test_loss, test_accuracy = model.evaluate(test_sequences, test_label)
    logging.info(f'Test Loss : {test_loss}')
    logging.info(f'Test Accuracy : {test_accuracy}')
    
   


############################
# General functions
############################

def init_logging():
  log_format = f"%(asctime)s [%(processName)s] [%(name)s] [%(levelname)s] %(message)s"
  log_level = logging.DEBUG
  if getattr(sys, 'frozen', False):
    folder = os.path.dirname(sys.executable)
  else:
    folder = os.path.dirname(os.path.abspath(__file__))
  # noinspection PyArgumentList
  logging.basicConfig(
    format=log_format,
    level=log_level,
    force=True,
    handlers=[
      logging.FileHandler(filename=os.path.join(folder, 'debug.log'), mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stdout)
    ])
  

def main():
    data = importData()
    clean_data = dataPreprocessing(data)
    train, test, train_label, test_label, tokenizer= tokenize(clean_data)
    model = train_model(train, test, train_label, test_label, tokenizer)
    evaluate(model, test, test_label)
    

if __name__ == '__main__':
    main()