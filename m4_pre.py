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
