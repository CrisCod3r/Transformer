from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

import tensorflow as tf

import keras
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split

def def_model(inp_shape = (50,50,3)):
    inp = Input(inp_shape)
    m = Conv2D(32, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')(inp)
    m = MaxPooling2D(2)(m)
    m = BatchNormalization()(m)
    m = Conv2D(64, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')(m)
    m = MaxPooling2D(2)(m)
    m = BatchNormalization()(m)
    m = Conv2D(128, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')(m)
    m = MaxPooling2D(2)(m)
    m = BatchNormalization()(m)
    m = Conv2D(128, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')(m)
    m = MaxPooling2D(2)(m)
    m = Flatten()(m)
    m = Dense(128, activation = "relu")(m)
    out = Dense(1, activation = "sigmoid")(m)
    model = Model(inp, out)
    model.compile(optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9), loss="binary_crossentropy", metrics = ['acc'])
    return model




samples = ImageDataGenerator(rescale = 1/255, validation_split=0.1)
train= samples.flow_from_directory(
    "../data_split/train/",
    # class_names=['0', '1']
    batch_size=32,
    target_size=(50, 50),  # reshape if not in this size
    shuffle=True,
    subset="training",
)

valid= samples.flow_from_directory(
    "../data_split/train/",
    # class_names=['0', '1']
    batch_size=32,
    target_size=(50, 50),  # reshape if not in this size
    shuffle=True,
    subset="validation",
)
for element in train:
    print(element)
    break
model = def_model()

# model.summary()

history = model.fit(train, validation_data=valid,  epochs = 15, steps_per_epoch=32, verbose=2)