import pickle
import math, re, os
from os import path

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import pandas as pd
import random
import shutil
import pathlib

EPOCHS = 150
BATCH_SIZE = 20

BASE_FOLDER = '.\\..\\data\\'

def create_cnn_model():
  my_output = layers.Dense(5, activation='softmax')
  model = MobileNetV2(weights='imagenet')

  x = model.layers[-2].output
  x = layers.Dense(575,activation='relu')(x)
  my_output = my_output(x)
  model = Model(inputs=model.input, outputs=my_output)
  for layer in model.layers[:-2]:
    layer.trainable = False


  model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy']
  )
  print(model.summary())
  return model

def create_callbacks():
  early_stopping = EarlyStopping(patience=35, monitor='val_loss', verbose=1)

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.001,
                                  patience=25, mode='min',
                                  verbose=1)

  model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath='.\\..\\models\\classification\\my_mobilenet_model.h5',
                                       save_best_only=True,
                                       verbose=1)

  callbacks = [
        early_stopping,
        reduce_lr,
        model_checkpoint
  ]

  return callbacks

def train_model_naive_split():

  train_gen = ImageDataGenerator(
        rescale=1./255 ,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
  )

  train_iterator = train_gen.flow_from_directory('.\\..\\data\\train\\',
                                                   target_size=(224, 224),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical')

  validation_gen = ImageDataGenerator(
      rescale=1./255 ,
      rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
      )
  validation_iterator = validation_gen.flow_from_directory('.\\..\\data\\valid\\',
                                                             target_size=(224, 224),
                                                             batch_size=BATCH_SIZE,
                                                             class_mode='categorical')

  model = create_cnn_model()

  history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        epochs=EPOCHS,
                        callbacks=create_callbacks())

  return history

history=train_model_naive_split()

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

def load_and_predict(model):
  
  test_generator = ImageDataGenerator(rescale=1. / 255)

  test_iterator = test_generator.flow_from_directory(
        '.\\..\\data\\test\\',
        target_size=(224, 224),
        shuffle=False,
        class_mode='categorical',
        batch_size=1) 

  ids = []
  for filename in test_iterator.filenames:
    print(filename)
    ids.append(filename)

  predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
  predictions = []
  for index, prediction in enumerate(predict_result):
    classes = np.argmax(prediction)
    predictions.append([ids[index], classes])
  predictions.sort()

  return predictions

def store_prediction():
  model = keras.models.load_model('.\\..\\models\\classification\\my_mobilenet_model.h5', compile = True)

  predictions = load_and_predict(model)

  df = pd.DataFrame(data=predictions, columns=['image_id', 'label'])
  df = df.set_index(['image_id'])

  print(df.head())
  print('Writing submission')
  df.to_csv('.\\..\\results\\CustomMobileNetV2Results.csv')

store_prediction()