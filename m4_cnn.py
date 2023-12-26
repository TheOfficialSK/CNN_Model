#    name: m4_cnn.py
# purpose: template for building a Keras model
#          for hand written number classification
#    NOTE: Submit a different python file for each model
# -------------------------------------------------


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.src.preprocessing.image import ImageDataGenerator

from m4_pre import processDataset
import argparse
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()


def main():
    np.random.seed(1671)

    parms = parseArguments()
    print("loading data...")
    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)
    print("Processing dataset...")
    X_train, y_train = processDataset(X_train, y_train)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2
    )

    print('KERA modeling build starting...')
    ## Build your model here

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    print('KERA modeling compiling...')
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

    print('KERA modeling fitting...')
    model.fit(datagen.flow(X_train, y_train, batch_size=750),
              epochs=14,
              steps_per_epoch=len(X_train) / 750,
              verbose=1)

    ## Evaluating the model
    print('KERA modeling evaluating...')
    _, accuracy = model.evaluate(X_train, y_train, verbose=1)
    print('Accuracy: %.2f' % (accuracy * 100))

    score = model.evaluate(X_train, y_train)

    print(model.metrics_names)

    print('Train score:', score[0], ' Train accuracy:', score[1])

    ## save your model
    model.save(parms.outModelFile)


if __name__ == '__main__':
    main()
