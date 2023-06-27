import logging
import os
import glob
import sys
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
#nltk.download('stopwords')
from email.parser import BytesParser
from email.policy import default
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

############################
# Train data
############################
def train(data: pd.DataFrame):
  try:
    # Split into features and label
    X = data.content
    y = data.label.astype(int)
    
    # Turn wordlist into tokens
    vector = CountVectorizer()
    X = vector.fit_transform(X)
    #X.toarray().shape
    
    # split into training and test data
    X_train,X_rem,y_train,y_rem = train_test_split(X, y, test_size=0.20, random_state=39)
    X_val,X_test,y_val,y_test = train_test_split(X_rem, y_rem, test_size=0.50, random_state=39)

    # KNN model
    knn = KNeighborsClassifier()
    model = knn.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    logging.info(f'"Accuracy for validation set: {(accuracy_score(y_val, y_pred)*100)}%')
  
  except Exception as e:
     print(e)


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
  count = 0
  for f in csv_files:
    for encoding in encodings:
      try:
        data = pd.read_csv(f,encoding=encoding, on_bad_lines='skip')
        spam_data.append(data)
        encoding_dict[f]=encoding
        count += 1
        logging.info(f'{count}/{len(csv_files)} processed.')
        break
      except Exception as e:  # or the error you receive
          pass
      
  # Enron Ham Data
  ham = pd.read_csv('./enron.csv')
  
  spam = pd.concat(spam_data)


  logging.info(f'{len(spam)} Spam sets and {len(ham)} valid sets.')
  data = pd.concat([ham[['email','label']], spam[['email','label']]], axis=0, ignore_index=True)
  return data


def main():
    #data = importData()
    #clean_data = dataPreprocessing(data)
    clean_data = pd.read_csv('./Data/SpamHamTokens.csv')
    train(clean_data)


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

if __name__ == '__main__':
    init_logging()
    main()