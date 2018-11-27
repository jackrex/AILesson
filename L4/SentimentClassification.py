#!/bin/env python
# -*- coding: utf-8 -*-
# pip3 install -U scikit-learn

from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas

MOVIE_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/train'
MOVIE_TEST_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/test'
train_movie_data = datasets.load_files(MOVIE_PATH, 'Movie Comments', ['neg', 'pos'], True, True, None, 'strict', 42)
test_movie_data = datasets.load_files(MOVIE_TEST_PATH, 'Movie Test Comments', ['neg', 'pos'], True, True, None,
                                    'strict', 42)
stop_words = pandas.read_csv("stopwords.txt", delimiter="\n", header=None)
print(stop_words.values)


def classifier_report(classifier):
    count_vec = CountVectorizer()
    x_train = count_vec.fit_transform(train_movie_data.data)
    print('x_train shape is ' + str(x_train.shape))
    print(x_train.todense())
    names = count_vec.get_feature_names()
    print(names)

    # 计算TF-IDF
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(x_train)  # 为词频矩阵的每个词加上权重（即TF * IDF），得到TF-IDF矩阵
    print(tfidf.todense())

    x_test = count_vec.transform(test_movie_data.data)
    print('x_test shape is ' + str(x_test.shape))

    classifier.fit(x_train, train_movie_data.target)

    predicted = classifier.predict(x_test)
    print(classification_report(test_movie_data.target, predicted, target_names=train_movie_data.target_names))
    print(metrics.confusion_matrix(test_movie_data.target, predicted))


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
    logistic_regression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None, solver='sag', max_iter=2000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)
    random_forest = RandomForestClassifier(n_estimators=8)
    decision_tree = DecisionTreeClassifier()

    # split_bayesian()
    # classifier_report(bayesian)
    classifier_report(logistic_regression)
    # classifier_report(random_forest)
    # classifier_report(decision_tree)



