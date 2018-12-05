#!/bin/env python
# -*- coding: utf-8 -*-
from openpyxl import load_workbook
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_words_data():
    wb = load_workbook(filename='Search_Index_20181118.xlsx')
    sheet = wb['index']
    words_data = []
    for row in sheet.rows:
        labels = []
        for cell in row:
            labels.append(cell.value)
        words_data.append(labels)
    words_label = words_data[0]
    words_data.pop(0)
    words_label.pop(0)

    # generate
    fix_words_data = []
    result_data = []
    for word_data in words_data:
        search_index = word_data[1]
        search_result = word_data[2]
        search_popular = word_data[3]
        app_name = word_data[4]

        if type(app_name) is int:
            continue

        fix_word_data = []
        if search_index > 4800:
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if search_result < 1000:
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if search_popular > 50:
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if "fitness" in app_name.lower() or "hiit" in app_name.lower():
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if search_index > 4800 and search_result < 1000 and search_popular > 40 and (
                "fitness" in app_name.lower() or "hiit" in app_name.lower() or "in" in app_name.lower() or "fi" in app_name.lower() ):
            result_data.append("success")
        else:
            result_data.append("failed")

        fix_words_data.append(fix_word_data)

    return result_data, fix_words_data

def load_iris():
    iris = datasets.load_iris()
    return iris.target, iris.data

def svm(k):
    labels, data = load_words_data()
    t = time.time()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=33)
    svc = SVC(kernel=k, C=1.0)
    svc.fit(x_train, y_train)
    y_predict = svc.predict(x_test)
    print('The result score is', svc.score(x_test, y_test))
    print(classification_report(y_test, y_predict))
    print('Time usage: ' + str(time.time() - t))


if __name__ == '__main__':
    kernels = ['linear', 'rbf', 'poly', 'sigmod', 'precomputed']
    for kernel in kernels:
        svm(kernel)



