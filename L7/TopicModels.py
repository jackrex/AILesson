#!/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV


MOVIE_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/train/'
train_movie_data = datasets.load_files(MOVIE_PATH, 'Movie Comments', ['unsup'], True, True, None, 'strict', 42)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print (model.components_)

def lda_train():
    count_vec = CountVectorizer(max_df=1.0, min_df=2,
                                stop_words='english')
    tf = count_vec.fit_transform(train_movie_data.data)
    words = count_vec.get_feature_names()
    tf_array = tf.toarray()

    print(words)
    print('weight = :' + str(len(tf.toarray())))

    result_dic = {}

    print("words---->count")
    for i in range(len(words)):
        key = "%s"%(words[i])
        result_dic[key] = 0
        for j in range(len(tf_array)):
            result_dic[key] += tf_array[j][i]

    result_dic = sorted(result_dic.items(), key=lambda d: d[1], reverse=True)
    top_500_dic = result_dic[1:501]

    print(top_500_dic)

    n_topics = 20
    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=500,
                                    learning_method='batch',
                                    random_state=0,
                                    n_jobs=-1)
    lda.fit(tf)
    doc_topic_dist = lda.transform(tf)
    print(doc_topic_dist)

    n_top_words = 20
    tf_feature_names = count_vec.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    print(lda.perplexity(tf))

def find_best_parameters():
    n_features = 250
    count_vec = CountVectorizer(max_df=1.0, min_df=5,
                                max_features=n_features,
                                stop_words='english')
    tf = count_vec.fit_transform(train_movie_data.data)

    parameters = {'learning_method': ('batch', 'online'),
                  'n_topics': range(20, 75, 5),
                  'perp_tol': (0.001, 0.01, 0.1),
                  'doc_topic_prior': (0.001, 0.01, 0.05, 0.1, 0.2),
                  'topic_word_prior': (0.001, 0.01, 0.05, 0.1, 0.2),
                  }

    lda = LatentDirichletAllocation()
    model = GridSearchCV(lda, parameters)
    model.fit(tf)
    sorted(model.cv_results_.keys())

if __name__ == '__main__':
    # lda_train()
    find_best_parameters()




