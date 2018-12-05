#!/bin/env python
# -*- coding: utf-8 -*-
# pip3 install -U scikit-learn
from scipy.stats import uniform
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import pandas
import time
import scipy

MOVIE_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/train'
MOVIE_TEST_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/test'
train_movie_data = datasets.load_files(MOVIE_PATH, 'Movie Comments', ['neg', 'pos'], True, True, None, 'strict', 42)
test_movie_data = datasets.load_files(MOVIE_TEST_PATH, 'Movie Test Comments', ['neg', 'pos'], True, True, None,
                                    'strict', 42)

stop_words = pandas.read_csv("stopwords.txt", delimiter="\n", header=None)
# print(stop_words.values)


def classifier_report(classifier):
    t = time.time()
    print(type(classifier))
    count_vec = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 2))
    x_train = count_vec.fit_transform(train_movie_data.data)
    # print('weight is ' + str(x_train.toarray()))
    # print(count_vec.get_feature_names())
    x_test = count_vec.transform(test_movie_data.data)

    classifier.fit(x_train, train_movie_data.target)

    predicted = classifier.predict(x_test)
    print(classification_report(test_movie_data.target, predicted, target_names=train_movie_data.target_names))
    print('Time usage: ' + str(time.time() - t))


def split_bayesian():
    print('split_bayesian')
    t = time.time()
    x_train, x_test, y_train, y_test = train_test_split(train_movie_data.data, train_movie_data.target, test_size=0.25, random_state=33)
    print(x_test)
    count_vec = CountVectorizer()
    x_count_train = count_vec.fit_transform(x_train)
    x_count_test = count_vec.transform(x_test)

    mnb_count = MultinomialNB()
    mnb_count.fit(x_count_train, y_train)
    y_count_predict = mnb_count.predict(x_count_test)
    print(classification_report(y_test, y_count_predict, target_names=train_movie_data.target_names))
    print('Time usage: ' + str(time.time() - t))


def optimize_logistic_regression():
    # classifier = LogisticRegression()
    classifier = LogisticRegression(penalty='l2', dual=False, tol=1e-6, C=1.0,fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)
    count_vec = CountVectorizer()
    x_train = count_vec.fit_transform(train_movie_data.data)
    print('x_train shape is ' + str(x_train.shape))
    print(x_train.todense())
    names = count_vec.get_feature_names()
    print(names)

    transformer = TfidfTransformer()
    tfidf_train = transformer.fit_transform(x_train)
    print(tfidf_train.todense())

    x_test = count_vec.transform(test_movie_data.data)
    tfidf_test = transformer.transform(x_test)
    print('x_test shape is ' + str(tfidf_test.shape))

    # tuned_parameters = {'C': uniform(loc=0, scale=4),
    #                     'multi_class': ['ovr', 'multinomial']}
    #
    # classifier = RandomizedSearchCV(LogisticRegression(penalty='l2', solver='lbfgs', tol=1e-6),
    #                          tuned_parameters, cv=10, scoring='accuracy', n_iter=30)



    # tuned_parameters = [{'penalty': ['l1', 'l2'],
    #                      'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    #                      'solver': ['liblinear'],
    #                      'multi_class': ['ovr']},
    #                     {'penalty': ['l2'],
    #                      'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
    #                      'solver': ['lbfgs'],
    #                      'multi_class': ['ovr', 'multinomial']}]
    #
    # classifier = GridSearchCV(LogisticRegression(tol=1e-6), tuned_parameters, cv=10, n_jobs=-1)

    classifier.fit(x_train, train_movie_data.target)

    # print('Best parameters set found:', classifier.best_params_)

    predicted = classifier.predict(x_test)
    print(classification_report(test_movie_data.target, predicted, target_names=train_movie_data.target_names))
    print(metrics.confusion_matrix(test_movie_data.target, predicted))


if __name__ == '__main__':

    # bayesian = MultinomialNB()
    # logistic_regression = LogisticRegression()
    # random_forest = RandomForestClassifier()
    # decision_tree = DecisionTreeClassifier()
    # logistic_cv = LogisticRegressionCV(cv=5, max_iter=100, n_jobs=-1)

    split_bayesian()

    # classifier_report(bayesian)
    # classifier_report(logistic_regression)
    # classifier_report(logistic_cv)
    # classifier_report(random_forest)
    # classifier_report(decision_tree)

    # optimize_logistic_regression()





