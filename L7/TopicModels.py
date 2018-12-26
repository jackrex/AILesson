#!/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from wordcloud import WordCloud
from scipy.misc import imread
from random import choice
import matplotlib.pyplot as plt
import pandas as pd
import sys




MOVIE_PATH = '/Users/jackrex/Desktop/AILesson/L4/aclImdb/train/'
train_movie_data = datasets.load_files(MOVIE_PATH, 'Movie Comments', ['neg'], True, True, None, 'strict', 42)

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

    print("词---->个数")
    for i in range(len(words)):
        key = "%s"%(words[i])
        result_dic[key] = 0
        for j in range(len(tf_array)):
            result_dic[key] += tf_array[j][i]

    # 对字典进行排序
    result_dic = sorted(result_dic.items(), key=lambda d: d[1], reverse=True)
    top_500_dic = result_dic[1:501]

    print(top_500_dic)
    return top_500_dic

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
    n_features = 2500
    count_vec = CountVectorizer(max_df=1.0, min_df=5,
                                max_features=n_features,
                                stop_words='english')
    tf = count_vec.fit_transform(train_movie_data.data)

    parameters = {'learning_method': ('batch', 'online'),
                  'n_topics': range(20, 75, 5),
                  'perp_tol': (0.001, 0.01, 0.1),
                  'doc_topic_prior': (0.001, 0.01, 0.05, 0.1, 0.2),
                  'topic_word_prior': (0.001, 0.01, 0.05, 0.1, 0.2),
                  'max_iter': 1000}

    lda = LatentDirichletAllocation()
    model = GridSearchCV(lda, parameters)
    model.fit(tf)
    sorted(model.cv_results_.keys())


def my_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return choice(["rgb(94,38,18)", "rgb(41,36,33)", "rgb(128,128,105)", "rgb(112,128,105)"])

def draw_cloud(mask_path, word_freq, save_path):
    mask = imread(mask_path)  #读取图片
    wc = WordCloud(background_color="white",
                   max_words=500,
                   mask=mask,
                   max_font_size=80,
                   random_state=42,
                   )
    wc.generate_from_frequencies(word_freq)

    plt.figure()

    plt.imshow(wc.recolor(color_func=my_color_func), interpolation='bilinear')

    plt.axis("off")
    wc.to_file(save_path)
    plt.show()

if __name__ == '__main__':
    top_dict = lda_train()
    d = {'items': top_dict}
    draw_cloud("./input.png", d, "./output.png")




