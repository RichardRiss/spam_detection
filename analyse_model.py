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
 
# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def main():
    model = tf.keras.models.load_model('./tensor_model')
    model: tf.keras.Model
    print(model.summary())
    




if __name__ == '__main__':
    main()