#!/bin/env python
# -*- coding: utf-8 -*-

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras_tqdm import TQDMCallback  # add progress
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import os

import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data

TRAIN_PATH = '/Users/jackrex/Desktop/AILesson/L8/训练集/'
TEST_PATH = '/Users/jackrex/Desktop/AILesson/L8/验证集/'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Build Model

batch_size = 1
train_data_gen = ImageDataGenerator(rescale=1 / 255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data_gen = ImageDataGenerator(rescale=1 / 255.)

train_generator = train_data_gen.flow_from_directory(TRAIN_PATH, classes=['其他', '卡通'], target_size=(150, 150), batch_size=batch_size,
                                                     class_mode="categorical")
test_generator = test_data_gen.flow_from_directory(TEST_PATH, target_size=(150, 150),classes=['其他', '卡通'], batch_size=batch_size,
                                                   class_mode="categorical")

labels = (train_generator.class_indices)
print(labels)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

epochs = 30
l_rate = 0.01
decay = l_rate / epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()


# Add Callback Function

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = LossHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


class SuccessHistory(Callback):
    def on_train_begin(self, logs={}):
        self.successes = []
        self.val_successes = []

    def on_epoch_end(self, epoch, logs=None):
        self.successes.append(logs.get('acc'))
        self.val_successes.append(logs.get('val_acc'))


success_history = SuccessHistory()
early_s_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='auto')

# Training

n = 30
ratio = 0.1
fit_model = model.fit_generator(train_generator, steps_per_epoch=int(n * (1 - ratio)), epochs=epochs,
                                validation_data=test_generator, validation_steps=int(n * ratio), callbacks=[success_history])

# test_generator.reset()
# pred = model.predict_generator(test_generator, verbose=1, steps=20)
# print(pred)
#
# predicted_class_indices = np.argmax(pred, axis=1)
# print(predicted_class_indices)
#
# filenames = test_generator.filenames
# for idx in range(len(filenames )):
#     print('predict  %s', ["其他", "卡通"][predicted_class_indices[idx]])
#     print('title    %s' % filenames[idx])
#     print('')

file_path = TEST_PATH + '其他/' + "006510_0.jpg"
img = image.load_img(file_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

y = model.predict(x)
print(y)


# dir_path = TEST_PATH + '其他'
# for pic_file in os.listdir(dir_path):
#     file_path = TEST_PATH + '其他/' + pic_file
#     img = image.load_img(file_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     y = model.predict(x)
#     print(y)
#     result = model.predict(x)


#
# losses, val_losses = success_history.successes, success_history.val_successes
# fig = plt.figure(figsize=(20, 5))
# plt.plot(fit_model.history['acc'], 'g', label='train accuracy')
# plt.plot(fit_model.history['val_acc'], 'r', label='val accuracy')
# plt.grid(True)
# plt.title('Training Accuracy vs Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
