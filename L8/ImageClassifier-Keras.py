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


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data

TRAIN_PATH = '/Users/jackrex/Desktop/AILesson/L8/训练集/'
TEST_PATH = '/Users/jackrex/Desktop/AILesson/L8/验证集/'

# Build Model

batch_size = 32
train_data_gen = ImageDataGenerator(rescale=1 / 255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data_gen = ImageDataGenerator(rescale=1 / 255.)

train_generator = train_data_gen.flow_from_directory(TRAIN_PATH, target_size=(150, 150), batch_size=batch_size,
                                                     class_mode="categorical")
test_generator = test_data_gen.flow_from_directory(TEST_PATH, target_size=(150, 150), batch_size=batch_size,
                                                   class_mode="categorical")

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

epochs = 20
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


# Training

n = 10
ratio = 0.1
fit_model = model.fit_generator(train_generator, steps_per_epoch=int(n * (1 - ratio)), epochs=epochs,
                                validation_data=test_generator, validation_steps=int(n * ratio),
                                callbacks=[TQDMCallback(), early_stopping, history])


losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fit_model.history['loss'], 'g', label='train losses')
plt.plot(fit_model.history['val_loss'], 'r', label='val losses')
plt.grid(True)
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
