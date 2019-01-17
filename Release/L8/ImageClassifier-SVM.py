#!/bin/env python
# -*- coding: utf-8 -*-

import skimage.color
import skimage.feature
import skimage.io
import skimage.transform
import sklearn.svm

import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data

TRAIN_PATH = '/Users/jackrex/Desktop/AILesson/L8/训练集/'
TEST_PATH = '/Users/jackrex/Desktop/AILesson/L8/验证集/'


def read_and_preprocess(im_path):
    im = skimage.io.imread(im_path)
    im = skimage.color.rgb2gray(im)
    im = skimage.transform.resize(im, (256, 256))
    return im


def get_data_tr():
    X = []
    Y = []
    for entry in os.scandir(TRAIN_PATH + '其他'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(0)
    for entry in os.scandir(TRAIN_PATH + '卡通'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(1)
    return X, Y


def get_data_te():
    X = []
    Y = []
    for entry in os.scandir(TEST_PATH + '其他'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(0)
    for entry in os.scandir(TEST_PATH + '卡通'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(1)
    return X, Y


# Train
Xtr, Ytr = get_data_tr()
clf = sklearn.svm.SVC(probability=True)
clf.fit(Xtr, Ytr)

# Test
Xte, Yte = get_data_te()
r = clf.predict(Xte)
s = 0
for i in range(len(r)):
    if r[i] == Yte[i]:
        s += 1
print('acc:', s / len(r))