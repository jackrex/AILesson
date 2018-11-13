#!/bin/env python
# -*- coding: utf-8 -*-
import os
import re, json

# Path
TRAIN_PATH = "20_newsgroups/"
TEST_DATA_PATH = "mini_newsgroups/"
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

feature_category_data = {}
category_count_data = {}
data_dict = {}


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
    pattern = re.compile(r'.*\d+')
    for i in range(len(letters)):
        # remove header
        if i == 0:
            continue

        if i == len(letters) - 1:
            # assume > 100 is good one, this letter not have signature
            if len(letters[i]) < 100:
                continue

            if "--" in letters[i]:
                continue

        text = letters[i].replace(",", "").replace(".", "").replace("*", "").replace("!", "").replace("?", "").replace(
            ">", "").replace(":", "").replace("-", "").replace("\"", "").replace("\n", " ").replace("\t", " ")
        words = text.split(" ")
        for word in words:
            # cast to lower case
            word = word.lower()

            # filter email & empty words
            if len(word) <= 1:
                continue

            # filter numbers
            if pattern.match(word) is not None:
                continue

            if word.find("@") != -1:
                continue

            if word.find("(") != -1:
                continue

            if word.find(")") != -1:
                continue

            if word.find("]") != -1:
                continue

            if word.find("[") != -1:
                continue

            if word in stopwords:
                continue

            news_words.append(word)

    return news_words


def load_training_data():
    if len(data_dict) > 0:
        return data_dict
    classify_names = os.listdir(TRAIN_PATH)
    if '.DS_Store' in classify_names:
        classify_names.remove('.DS_Store')
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
            for word in words:
                category_count_data[data_key] += 1
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
    basic_prob = float(numerator) / denominator

    # avoid zero
    if basic_prob == 0:
        return float(1.0)/100000


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
    category_prob = 1
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


def test_mini_training_prob():
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

        print (test_name + " prob is :  " + str(float(i) / 100))


def train_and_test_data():
    file_size = int(0.7 * len(data_dict['talk.politics.mideast']))
    train_data = {}
    test_data = {}
    for key in data_dict.keys():
        data_list = data_dict[key]
        train_data.setdefault(key, data_list[:file_size])

    test_names = os.listdir(TRAIN_PATH)
    if '.DS_Store' in test_names:
        test_names.remove('.DS_Store')
    for test_name in test_names:
        news_dir = CURRENT_PATH + "/" + TRAIN_PATH + test_name
        file_names = os.listdir(news_dir)
        paths = []
        for letter_file in file_names[file_size:]:
            paths.append(CURRENT_PATH + "/" + TRAIN_PATH + test_name + "/" + letter_file)
        test_data.setdefault(test_name, paths)

    return train_data, test_data


def test_shuffle_training_data_prob():
    load_training_data()
    train_data, test_data = train_and_test_data()
    training_data(train_data)
    for test_name in test_data.keys():
        i = 0
        for letter_file in test_data[test_name]:
            result = classify_doc(letter_file)
            if result == test_name:
                i += 1

        print (test_name + " prob is :  " + str(float(i) / len(test_data[test_name])))


if __name__ == '__main__':

     # data = {"good":[["hello", "hello", "hi","how","you","best"],["hello","best"]], "bad": [["fuck","jj","not","fu","you","yy"],["fuck","jj"]]}
     # print(training_data(data))

     test_shuffle_training_data_prob()

    # feature_category_data = training_data(load_training_data())
    # for word in feature_category_data.keys():
    #     count = 0
    #     for cate_count in feature_category_data[word].keys():
    #       count += feature_category_data[word][cate_count]
    #
    #     if count > 500:
    #         print (word + '------' + json.dumps(feature_category_data[word]))


    # Test mini news groups in all news groups
    # test_mini_training_prob()

    # Test single one
    # path = CURRENT_PATH + "/" + DATA_PATH + "soc.religion.christian/" + "20629"
    # result = classify_doc(path)
    # print result
