#    name: pa2pre.py
# purpose: Student's add code to preprocessing of the data

# Recall that any preprocessing you do on your training
# data, you must also do on any future data you want to
# predict.  This file allows you to perform any
# preprocessing you need on my undisclosed test data

import numpy as np
from keras import utils
from keras.preprocessing.image import ImageDataGenerator


def processDataset(X, y):
    X = X.astype('float32') / 255
    X = X.reshape(X.shape[0], 28, 28, 1)
    y = utils.to_categorical(y, 10)
    return X, y


def processTestData(X, y):
    X = X.astype('float32') / 255
    X = X.reshape(X.shape[0], 28, 28, 1)
    y = utils.to_categorical(y, 10)
    return X, y
