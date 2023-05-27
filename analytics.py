# -*- coding: cp1251 -*-

import re
import json
import math
import pandas as pd
import numpy as np
import jarowinkler 

STD_CATEGORIES = {
    0 : ["PickPoint", "ПикПоинт"],
    1 : ["Ozon", "Озон"],
    2 : ["Сбермегамаркет", "Sbermegamarket"],
    3 : ["Wildberries", "Вайлберис"],
    4 : ["Яндекс.Маркет", "Яндекс", "Yandex", "Маркет", "Market"],
    5 : ["Boxberry", "Боксберри"],
    6 : ["QIWI", "КИВИ"],
    7 : ["Халва", "Halva"],
    
}

STD_CATEGORIESLiST = [
    "PickPoint",
    "Ozon",
    "Сбермегамаркет",
    "Wildberries",
    "Яндекс.Маркет",
    "Boxberry",
    "QIWI",
    "Халва",
    
]


def filtered_split(input_text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' ', outlen=50):

    input_text = input_text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    input_text = input_text.translate(translate_map)

    seq = input_text.split(split)
    return [i for i in seq if i]


def define_category(texts, categoryies):

    result = []

    for text in texts:

        key, name = define_category_text(text, categoryies)

        result += [{"Код": key, "Название": name}]

    return result


def define_category_text(text, categoryies):

    words = filtered_split(text)

    for word in words:

        for key in categoryies:
            for keyword in categoryies[key]:

                value = jarowinkler.jarowinkler_similarity(word, keyword.lower())

                if 0.80 < value:
                    return key, STD_CATEGORIESLiST[key]

    return -1, ""





