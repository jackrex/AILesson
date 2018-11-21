#!/bin/env python
# -*- coding: utf-8 -*-
from math import log, exp
from openpyxl import load_workbook
import operator
import json
import time
import random


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
    for word_data in words_data:
        keyword = word_data[0]
        search_index = word_data[1]
        search_result = word_data[2]
        search_popular = word_data[3]
        app_name = word_data[4]

        if type(app_name) is int:
            continue

        fix_word_data = []
        if search_index > 4800:
            fix_word_data.append("valid search index")
        else:
            fix_word_data.append("invalid search index")

        if search_result < 1000:
            fix_word_data.append("valid search result")
        else:
            fix_word_data.append("invalid search result")

        if search_popular > 50:
            fix_word_data.append("popular")
        else:
            fix_word_data.append("unpopular")

        if "fitness" in app_name.lower() or "hiit" in app_name.lower():
            fix_word_data.append("valid words")
        else:
            fix_word_data.append("invalid words")

        if search_index > 4800 and search_result < 1000 and search_popular > 40 and (
                "fitness" in app_name.lower() or "hiit" in app_name.lower() or "in" in app_name.lower() ):
            print ('keyword is ' + keyword)
            fix_word_data.append("success")
        else:
            fix_word_data.append("failed")

        fix_words_data.append(fix_word_data)

    return words_label, fix_words_data


def cal_shannon_entropy(data_set):
    num = len(data_set)
    label_counts = {}
    for data in data_set:
        current_label = data[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0
    for key in label_counts:
        prob = float(label_counts[key]) / num
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def choose_best_feature(words_data):
    feature_num = len(words_data[0]) - 1
    base_entropy = cal_shannon_entropy(words_data)
    print('base_entropy is : ' + str(base_entropy))
    best_info_gain = 0
    best_feature = -1
    for i in range(feature_num):
        feature_list = [feature[i] for feature in words_data]
        new_entropy = 0
        for value in set(feature_list):
            sub_data = []
            for words in words_data:
                if words[i] == value:
                    sub_vec = words[:i]
                    sub_vec.extend(words[i+1:])
                    sub_data.append(sub_vec)
            prob = len(sub_data) / float(len(words_data))
            new_entropy += prob * cal_shannon_entropy(sub_data)
        print('new_entropy is + ' + str(new_entropy))
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def create_tree(labels, words_data):
    result_list = [example[-1] for example in words_data]
    if result_list.count(result_list[0]) == len(result_list):
        return result_list[0]
    if len(words_data[0]) == 1:
        # vote for result
        class_count = {}
        for vote in result_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        sorted_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_count[0][0]
    best_feature = choose_best_feature(words_data)
    best_feature_label = labels[best_feature]
    decision_tree = {best_feature_label: {}}
    del (labels[best_feature])
    feature_values = [example[best_feature] for example in words_data]
    for value in set(feature_values):
        sub_labels = labels[:]
        sub_data = []
        for words in words_data:
            if words[best_feature] == value:
                sub_vec = words[:best_feature]
                sub_vec.extend(words[best_feature + 1:])
                sub_data.append(sub_vec)
        decision_tree[best_feature_label][value] = create_tree(sub_labels, sub_data)
    return decision_tree


def create_train_test_data(all_data, label):
    size = int(len(all_data) * 0.7)
    t_train_data = all_data[:size]
    t_test_data = all_data[size:]
    t = []
    for test in t_test_data:
        t_dict = {}
        for la in range(len(label)):
            t_dict.setdefault(label[la], test[la])
        t.append(t_dict)

    return t_train_data, t


def convert_test_data(t_test_data, label):
    t = []
    for test in t_test_data:
        dict = {}
        for la in range(len(label)):
            dict.setdefault(label[la], test[la])
        t.append(dict)
    return t


def valid_data(test_data, tree):
    valid = 0
    for test in test_data:
        if test['Search-Result'] == 'valid search result' and test['Search-Index'] == 'valid search index':
            if test['Result'] == 'success':
                valid+1
        else:
            if test['Result'] == 'failed':
                valid += 1

    print('prob is ' + str(float(valid)/len(test_data)))
    return float(valid)/len(test_data)

def valid_f_data(test_data, tree):
    valid = 0
    for test in test_data:
        if test['App-Name'] == 'valid words':
            if test['Result'] == 'success':
                valid+=1
        else:
            if test['Result'] == 'failed':
                valid += 1

    print('prob is ' + str(float(valid)/len(test_data)))
    return float(valid)/len(test_data)


def cross_validation_split_for_random_forest(data_set, n):
    data_set_split = list()
    data_set_copy = list(data_set)
    fold_size = int(len(data_set) / n)
    for i in range(n):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(data_set_copy))
            fold.append(data_set_copy.pop(index))
        data_set_split.append(fold)
    return data_set_split


def cal_random_forest_prob(all_data, label):
    datas = cross_validation_split_for_random_forest(all_data, 10)
    probs = []
    for data in datas:
        l = list(label)
        train_data = list(datas)
        train_data.remove(data)
        da = []
        for d in train_data:
            da += d
        test_data = data
        td = convert_test_data(test_data, l)
        tree = create_tree(l, da)
        print(tree)
        prob = valid_f_data(td, tree)
        probs.append(prob)

    print('max is ' + str(max(probs)))

def cal_alpha(error):
    alpha = 1 / 2 * log((1 - error) / error)
    return alpha


def cal_new_weight(alpha, weight):
    new_weight = []
    sum_weight = 0
    for i in range(len(weight)):
        flag = 1
        weighti = weight[i] * exp(-alpha * flag)
        new_weight.append(weighti)
        sum_weight += weighti
    return new_weight


def train_ada_boost(label, data, error_threshold, max_iter):
    fx = {}
    weight = []
    num = len(label)
    for i in range(num):
        w = float(1 / num)
        weight.append(w)

    for i in range(max_iter):
        fx[i] = {}
        td = convert_test_data(data, label)
        prob = valid_data(td, labels)
        error = (weight[i] * prob)
        alpha = cal_alpha(error)
        weight = cal_new_weight(alpha, weight)
        fx[i]['alpha'] = alpha / num
        if error < error_threshold:
            break
    return fx



if __name__ == '__main__':
    labels, data = load_words_data()
    l = list(labels)
    # random.shuffle(data)

    train_data, test_data = create_train_test_data(data, labels)
    t = time.time()
    tree = create_tree(labels, train_data)
    valid_data(test_data, tree)
    print('time gap is:' + str(time.time() - t) )
    print (tree)

    t1 = time.time()
    cal_random_forest_prob(data, l)
    print('time t1 gap is:' + str(time.time() - t1) )

    ada = train_ada_boost(l, data, 0.01, 5)
    print(ada)


    # with open('tree.json', 'w') as outfile:
    #     json.dump(tree, outfile)

