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

