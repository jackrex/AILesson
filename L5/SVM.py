#!/bin/env python
# -*- coding: utf-8 -*-
from openpyxl import load_workbook
from sklearn.svm import SVC
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

def load_digits():
    digits = datasets.load_digits()
    return digits.target, digits.data

def svm(k):
    print('=======' + k + '=======')
    labels, data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=33)
    clf = SVC(kernel=k, C=10,gamma=0.7,degree=3)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_predict = clf.predict(x_test)
    print('The result score is', clf.score(x_test, y_test))
    # print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    svm('GridSearch')

    # kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    # for kernel in kernels:
    #     svm(kernel)








