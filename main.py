import logging
import os
import sys
import pandas as pd
import numpy as np
import sklearn

def importData():
  spam_ham = pd.read_csv('./Spam_Ham_data.csv')
  spam = spam_ham.loc[spam_ham.label == 1.0]
  ham = spam_ham.loc[spam_ham.label == 0]
  logging.info(f'{len(spam)} Spam sets and {len(ham)} valid sets.')


def main():
    importData()

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