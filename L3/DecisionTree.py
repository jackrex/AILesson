#!/bin/env python
# -*- coding: utf-8 -*-

from openpyxl import load_workbook
import math
import operator


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

        if type(app_name) is long:
            continue

        fix_word_data = []
        if search_index > 4800:
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if search_result < 1000:
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if search_popular > 40:
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if "fitness" in app_name.lower() or "hiit" in app_name.lower():
            fix_word_data.append(1)
        else:
            fix_word_data.append(0)

        if search_index > 4800 and search_result < 1000 and search_popular > 40 and (
                "fitness" in app_name.lower() or "hiit" in app_name.lower()):
            print 'keyword is ' + keyword
            fix_word_data.append(True)
        else:
            fix_word_data.append(False)

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
        prob = float(label_counts[key]) / num  # 计算单个类的熵值
        shannon_entropy -= prob * math.log(prob, 2)  # 累加每个类的熵值
    return shannon_entropy


def choose_best_feature(labels, words_data):
    feature_num = len(labels) - 1
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
                    sub_data = words[:i]
                    sub_data.extend(words[i+1:])
            prob = len(sub_data) / float(len(words_data))
            new_entropy += prob * cal_shannon_entropy(sub_data)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def create_tree(labels, words_data):
    result_list = [example[-1] for example in words_data]  # 类别：男或女
    if result_list.count(result_list[0]) == len(result_list):
        return result_list[0]
    if len(words_data[0]) == 1:
        # vote for result
        for vote in result_list:
            result_list[vote] += 1
        sorted_count = sorted(result_list.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_count[0][0]
    best_feature = choose_best_feature(labels, words_data)
    best_feature_label = labels[best_feature]
    decision_tree = {best_feature_label: {}}  # 分类结果以字典形式保存
    del (labels[best_feature_label])
    feature_values = [example[best_feature] for example in words_data]
    for value in set(feature_values):
        sub_labels = labels[:]
        sub_data = []
        for words in words_data:
            if words[best_feature] == value:
                sub_data = words[:best_feature]
                sub_data.extend(words[best_feature + 1:])
        decision_tree[best_feature_label][value] = create_tree(sub_labels, sub_data)
    return decision_tree


if __name__ == '__main__':
    labels, data = load_words_data()
    tree = create_tree(labels, data)
    print tree
