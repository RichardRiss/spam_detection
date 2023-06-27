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
import nltk
from nltk.corpus import stopwords
import re

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from email.parser import BytesParser
from email.policy import default



############################
# Import Data
############################

def importData():
      
  data = pd.read_csv('./Data/yahoo.csv',names=['email','label'],sep=';')
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




def main():
    data = importData()
    clean_data = dataPreprocessing(data)
    model = tf.keras.models.load_model('./tensor_model/spam_detection')
    model: tf.keras.Model

    X = clean_data.content
    test_labels = clean_data.label.astype(int)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)
    max_len = 100  # maximum sequence length
    test_sequences = pad_sequences(sequences,
                                    maxlen=max_len,
                                    padding='post',
                                    truncating='post')
    
    eval = model.evaluate(test_sequences,test_labels)
    print(eval)
    print("Stop")  
    
    




if __name__ == '__main__':
    main()