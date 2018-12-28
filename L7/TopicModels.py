#!/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
import time
import numpy
import os
import shutil

MOVIE_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/train/'
train_movie_data = datasets.load_files(MOVIE_PATH, 'Movie Comments', ['unsup'], True, True, None, 'strict', 42)
split_data = datasets.load_files(MOVIE_PATH + 'split/', 'Topics', None, True, True, None, 'strict', 42)

def load_data_vector():
    count_vec = CountVectorizer(max_df=1.0, min_df=2,
                                max_features=1000,
                                stop_words='english')
    tf = count_vec.fit_transform(train_movie_data.data)
    words = count_vec.get_feature_names()
    tf_array = tf.toarray()

    # result_dic = {}
    # print("words---->count")
    # for i in range(len(words)):
    #     key = "%s"%(words[i])
    #     result_dic[key] = 0
    #     for j in range(len(tf_array)):
    #         result_dic[key] += tf_array[j][i]
    #
    # result_dic = sorted(result_dic.items(), key=lambda d: d[1], reverse=True)
    # top_500_dic = result_dic[0:500]
    #
    # print(top_500_dic)
    return tf, count_vec

def lda_train():
    tf, count_vec = load_data_vector()
    n_topics = 20
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=10,
                                    learning_method='batch',
                                    random_state=0,
                                    perp_tol=0.01,
                                    topic_word_prior=0.2,
                                    n_jobs=-1)
    lda.fit(tf)
    doc_topic_dist = lda.transform(tf)
    print(doc_topic_dist)

    n_top_words = 20
    tf_feature_names = count_vec.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    print(lda.perplexity(tf))

def split_unsup_data():
    tf, count_vec = load_data_vector()
    n_topics = 20
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=10,
                                    learning_method='batch',
                                    random_state=0,
                                    perp_tol=0.01,
                                    topic_word_prior=0.2,
                                    n_jobs=-1)
    lda.fit(tf)
    doc_topic_dist = lda.transform(tf)
    print(doc_topic_dist)
    topics = []
    for doc_topic_probs in doc_topic_dist:
        tmp_array = numpy.array(doc_topic_probs)
        bb = tmp_array.tolist()
        index = bb.index(max(bb))
        topics.append(index)
        print('#Topic is #' + str(index))

    for i in range(0, 20):
        path = MOVIE_PATH  + 'split/' + 'Topic' + str(i)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    for i in range(0, 50000):
        file_move_path = MOVIE_PATH  + 'split/' + 'Topic' + str(topics[i])
        file_origin_path = MOVIE_PATH + 'unsup/' + str(i) + '_0.txt'
        shutil.copy(file_origin_path, file_move_path)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print (model.components_)

def find_best_parameters():
    count_vec = CountVectorizer(max_df=1.0, min_df=2,
                                max_features=1000,
                                stop_words='english')
    tf = count_vec.fit_transform(train_movie_data.data)

    parameters = {'learning_method': ('batch', 'online'),
                  'n_components': range(20, 75, 5),
                  'perp_tol': (0.001, 0.01, 0.1),
                  'topic_word_prior': (0.001, 0.01, 0.05, 0.1, 0.2),
                  }

    lda = LatentDirichletAllocation()
    model = GridSearchCV(lda, parameters)
    model.fit(tf)
    print(model.best_params_)

def classifier_valid():
    bayesian = MultinomialNB()
    logistic_regression = LogisticRegression()
    random_forest = RandomForestClassifier()
    decision_tree = DecisionTreeClassifier()

    classifier_report(bayesian)
    classifier_report(logistic_regression)
    classifier_report(random_forest)
    classifier_report(decision_tree)

def classifier_report(classifier):
    t = time.time()
    print(type(classifier))
    X_train, X_test, Y_train, Y_test = train_test_split(split_data.data, split_data.target, test_size=0.3, random_state=33)
    # print('weight is ' + str(x_train.toarray()))
    # print(count_vec.get_feature_names())

    count_vec = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 2))
    x_train = count_vec.fit_transform(X_train)
    # print('weight is ' + str(x_train.toarray()))
    # print(count_vec.get_feature_names())
    x_test = count_vec.transform(X_test)

    classifier.fit(x_train, Y_train)
    print(classifier.score(x_test, Y_test))
    print('Time usage: ' + str(time.time() - t))


if __name__ == '__main__':
    # lda_train()
    # find_best_parameters()
    # split_unsup_data()
    classifier_valid()




