#!/bin/env python
# -*- coding: utf-8 -*-
# pip3 install -U scikit-learn

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

MOVIE_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/train'
MOVIE_TEST_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/test'
train_movie_data = datasets.load_files(MOVIE_PATH, 'Movie Comments', ['neg', 'pos'], True, True, None, 'strict', 42)
test_movie_data = datasets.load_files(MOVIE_TEST_PATH, 'Movie Test Comments', ['neg', 'pos'], True, True, None,
                                    'strict', 42)

def classifier_report(classifier):
    count_vec = CountVectorizer()
    x_train = count_vec.fit_transform(train_movie_data.data)

    x_test = count_vec.transform(test_movie_data.data)

    classifier.fit(x_train, train_movie_data.target)

    predicted = classifier.predict(x_test)
    print(classification_report(test_movie_data.target, predicted, target_names=train_movie_data.target_names))

def split_bayesian():
    x_train, x_test, y_train, y_test = train_test_split(train_movie_data.data, train_movie_data.target, test_size=0.25, random_state=33)
    count_vec = CountVectorizer()
    x_count_train = count_vec.fit_transform(x_train)
    x_count_test = count_vec.transform(x_test)

    mnb_count = MultinomialNB()
    mnb_count.fit(x_count_train, y_train)
    y_count_predict = mnb_count.predict(x_count_test)
    print(classification_report(y_test, y_count_predict, target_names=train_movie_data.target_names))


if __name__ == '__main__':
    bayesian = MultinomialNB()
    logistic_regression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None, solver='sag', max_iter=10000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=10)

    # classifier_report(bayesian)
    # split_bayesian()
    classifier_report(logistic_regression)



