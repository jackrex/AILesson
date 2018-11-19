#!/bin/env python
# -*- coding: utf-8 -*-
from math import log
from openpyxl import load_workbook
import operator
import json


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

        if search_popular > 40:
            fix_word_data.append("popular")
        else:
            fix_word_data.append("unpopular")

        if "fitness" in app_name.lower() or "hiit" in app_name.lower():
            fix_word_data.append("valid words")
        else:
            fix_word_data.append("invalid words")

        if search_index > 4800 and search_result < 1000 and search_popular > 40 and (
                "fitness" in app_name.lower() or "hiit" in app_name.lower()):
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
    decision_tree = {best_feature_label: {}}  # 分类结果以字典形式保存
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


if __name__ == '__main__':
    labels, data = load_words_data()
    tree = create_tree(labels, data)
    print (tree)
    # with open('tree.json', 'w') as outfile:
    #     json.dump(tree, outfile)

