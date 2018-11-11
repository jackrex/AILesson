#!/bin/env python
# -*- coding: utf-8 -*-
import os

# Path
TRAIN_PATH = "20_newsgroups/"
TEST_DATA_PATH = "mini_newsgroups/"
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

feature_category_data = {}
category_count_data = {}

# Classify Category
# talk.politics.mideast
# rec.autos
# comp.sys.mac.hardware
# alt.atheism
# rec.sport.baseball
# comp.os.ms-windows.misc
# rec.sport.hockey
# sci.crypt
# sci.med
# talk.politics.misc
# rec.motorcycles
# comp.windows.x
# comp.graphics
# comp.sys.ibm.pc.hardware
# sci.electronics
# talk.politics.guns
# sci.space
# soc.religion.christian
# misc.forsale
# talk.religion.misc


def get_words(doc_path):
    news_letter = open(doc_path)
    letters = news_letter.read().split("\n\n")
    news_words = []
    stopwords = map(lambda s: s.strip(), open("stopwords.txt").readlines())
    for i in range(len(letters)):
        if i == 0:
            continue
        text = letters[i].replace(",", "").replace(".", "").replace(">", "").replace("\n", " ").replace("\t", " ")
        words = text.split(" ")
        for word in words:
            # filter email & empty words
            if len(word) == 0:
                continue

            if word.find("@") != -1:
                continue

            if word in stopwords:
                continue

            news_words.append(word)

    return news_words


def load_training_data():
    classify_names = os.listdir(TRAIN_PATH)
    if '.DS_Store' in classify_names:
        classify_names.remove('.DS_Store')
    data_dict = {}
    for classify_name in classify_names:

        # Test
        # if classify_name != "alt.atheism":
        #     continue

        news_dir = CURRENT_PATH + "/" + TRAIN_PATH + "/" + classify_name
        file_names = os.listdir(news_dir)
        all_words = []
        for letter_file in file_names:
            news_words = get_words(news_dir + "/" + letter_file)
            all_words.append(news_words)

        data_dict.setdefault(classify_name, all_words)
    return data_dict


def training_data(data):
    for data_key in data.keys():
        list_data = data[data_key]
        category_count_data.setdefault(data_key, 0)
        for words in list_data:
            category_count_data[data_key] += 1
            for word in words:
                feature_category_data.setdefault(word, {})
                feature_category_data[word].setdefault(data_key, 0)
                feature_category_data[word][data_key] += 1

    return feature_category_data


def word_in_category(word, category):
    if word in feature_category_data and category in feature_category_data[word]:
        return feature_category_data[word][category]
    else:
        return 0


def category_count(category):
    if category in category_count_data:
        return category_count_data[category]
    return 0


def total_count():
    return sum(category_count_data.values())


def prob_word_in_category(word, category):
    if category_count_data[category] == 0:
        return 0
    numerator = word_in_category(word, category)
    denominator = category_count(category)
    total = 0
    for cate in category_count_data.keys():
        total += word_in_category(word, cate)
    basic_prob = float(numerator) / denominator
    # avoid zero
    return (0.5 + total * basic_prob) / (1 + total)


def prob_doc_in_category(doc, category):
    words = get_words(doc)
    p = 1
    for word in words:
        p *= prob_word_in_category(word, category)
    return p


def prob_category_in_doc(doc, category):
    # p(category | document) = p(document | category) * p(category) / p(document)
    # p(document) = 1 or ignore
    # result =  p(document | category) * p(category)
    category_prob = category_count(category) / float(total_count())
    category_doc_prob = category_prob * prob_doc_in_category(doc, category)
    return category_doc_prob


def classify_doc(doc):
    max_value = 0.0
    best = ""
    for category in category_count_data.keys():
        prob = prob_category_in_doc(doc, category)
        # print (category, prob)
        if prob > max_value:
            max_value = prob
            best = category

    return best


def test_training_prob():
    test_names = os.listdir(TEST_DATA_PATH)
    if '.DS_Store' in test_names:
        test_names.remove('.DS_Store')
    for test_name in test_names:
        news_dir = CURRENT_PATH + "/" + TEST_DATA_PATH + test_name
        file_names = os.listdir(news_dir)
        i = 0
        for letter_file in file_names:
            result = classify_doc(news_dir + "/" + letter_file)
            if result == test_name:
                i += 1

        print (test_name + "prob is :  " + str(float(i)/100))


if __name__ == '__main__':

    training_data(load_training_data())
    test_training_prob()
    # Test single one
    # path = CURRENT_PATH + "/" + DATA_PATH + "soc.religion.christian/" + "20629"
    # result = classify_doc(path)
    # print result


