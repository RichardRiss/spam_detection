# General stuff
import logging
import os
import sys
import glob
import time
import json

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

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import brier_score_loss
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer




############################
# Import Data
############################

def importData():
  sample_data = pd.read_csv('./Data/sample.csv')
  data = pd.read_csv('./Data/yahoo.csv',names=['email','label'],sep=';')
  

  return data,sample_data



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
    text = re.sub(r"b'",'',text)
    text = re.sub(r"b\'",'',text)
    text = re.sub(r"b\\'",'',text)
    return text


def stopword(text):
    return ' '.join([w for w in text.split() if w not in stop_words])


def dataPreprocessing(data: pd.DataFrame):
  data['content'] = None
  for i in range(data.shape[0]):
    email = BytesParser(policy=default).parsebytes(data['email'][i].encode('utf-8'))
    payload = email.get_payload()
    if payload == '':
       payload = data['email'][i].encode('utf-8') 
    data['content'][i] = stopword(clean_text(str(payload)))
  
  data = data.dropna(axis=0)
  data = data.reset_index(drop=True)
  return data


def evaluate(model, test_sequences,tres = 95):
    prediction = model.predict(test_sequences)
    reconstructed_sequences = np.argmax(prediction, axis=-1)
    reconstruction_errors = np.mean(np.square(test_sequences - reconstructed_sequences), axis=1)

    # Plot the distribution of reconstruction errors
    plt.hist(reconstruction_errors, bins=50)
    plt.xlabel('Reconstruction error')
    plt.ylabel('No of examples')
    #plt.show()

    # Choose a threshold value that separates normal emails from anomalies
    threshold = np.percentile(reconstruction_errors, tres)  # adjust this value based on your plot

    # Classify the test examples as normal or anomaly based on the reconstruction error
    test_predictions = (reconstruction_errors > threshold).astype(int)

    return test_predictions,reconstruction_errors 



def main():
    '''
    hist = json.load(open('./tensor_model/history','r'))
    #plt.plot(hist['accuracy'])
    plt.plot(hist['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.show()
    '''

    data,sample_data = importData()
    #vectorizer = TfidfVectorizer()  # Initialize the TF-IDF vectorizer
    vectorizer = joblib.load('./forest_model/vectorizer.joblib')

    data_tfidf = vectorizer.transform(data['email'])  # Vectorize the training data
    
    #clean_data = dataPreprocessing(data)
    #clean_sample = dataPreprocessing(sample_data)
    model = tf.keras.models.load_model(filepath='./tensor_model/spam_detection_autoencoder3')
    model: (tf.keras.Model|None)

    #model = joblib.load('./forest_model/random_forest.joblib')


    #X = clean_data.content
    labels = data.label.astype(int)
    data = dataPreprocessing(data)
    data = data['content']
    vocab_size=100
    oov_tok='<OOV>'
    tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

    # Yahoo Data
    tokenizer.fit_on_texts(data)
    test_sequences = tokenizer.texts_to_sequences(data)
    max_len = 100  # maximum sequence length

    test_sequences = pad_sequences(test_sequences,
                                    maxlen=max_len,
                                    padding='post',
                                    truncating='post')
    
    
    # Sample Data
    '''
    sample_labels = sample_data.label.astype(int)
    tokenizer.fit_on_texts(clean_sample.content)
    sample_sequences = tokenizer.texts_to_sequences(clean_sample.content)
    max_len = 100  # maximum sequence length
    sample_sequences = pad_sequences(sample_sequences,
                                    maxlen=max_len,
                                    padding='post',
                                    truncating='post')
  
    '''

    # Evaluate
    labels = labels.to_numpy()
    labels_inv = np.where((labels==0)|(labels==1), labels^1, labels)
    t = time.time()
    eval,rec_error = evaluate(model,test_sequences)
    #eval,rec_error =  evaluate(model, test_sequences)
    t_func = time.time() - t
    
    #y_score = model.predict_proba(data_tfidf)[:,1]


    #fpr, tpr, _ = roc_curve(labels_inv, y_score)
    #roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    '''
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''



    error = np.mean( eval != labels )

    # ROC
    '''
    fpr, tpr, _ = roc_curve(labels_inv, rec_error)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2  # line width
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    '''

    # TP,FP,FN,TN
    # Calculating TP, FP, TN, FN
    TP = np.sum((labels_inv == 1) & (eval == 1))
    FP = np.sum((labels_inv == 0) & (eval == 1))
    TN = np.sum((labels_inv == 0) & (eval == 0))
    FN = np.sum((labels_inv == 1) & (eval == 0))

    # Categorizing data
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    values = [TP, FP, TN, FN]

    # Creating a DataFrame for seaborn plot
    df = pd.DataFrame(list(zip(categories*len(labels_inv), np.repeat(values, len(labels_inv)))), 
                      columns=['Categories', 'Values'])

    # Plotting the data
    plt.figure(figsize=(10, 8))
    sns.swarmplot(x="Categories", y="Values", data=df, s=10, color=".25")
    sns.boxplot(x="Categories", y="Values", data=df, showfliers=False)

    plt.title('Classification Results')
    plt.ylabel('Counts')
    plt.show()


    bs = brier_score_loss(labels,eval)
    print(f'Das Model erreicht folgenden Brier Score {bs} .')
    print(f'Modellbearbeitungszeit {round(t_func,4)} s.')
    print("Stop")  
    


if __name__ == '__main__':
    main()