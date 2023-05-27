# -*- coding: cp1251 -*-



import flask

import os
import json
import requests

import AI
import analytics

BASENAME = os.path.basename(__file__)
PATH = os.path.abspath(__file__).replace(BASENAME, "")

tokenizer = AI.PreTokenizer(PATH + "AI_data\\encoder\\encoder.w2v")
moderation = AI.ModerationHelper(PATH + "AI_data\\Moderation\\", tokenizer)
color = AI.ColorHelper(PATH + "AI_data\\Color\\", tokenizer)

app = flask.Flask(__name__)



@app.route("/", methods=["GET"])
def _qweqw():
    return json.dumps({"result": 0, "message": "Работает."})


"""
    POST /moderation/texts
    json data:
    { data: ["...", "...", ...] }

    output:
    {
        result: 0 - успех, иначе ошибка
        message и python_message описание ошибки
        data: [
            {
                Статус: 0 / 1
                Комментарий
                Подробно: {
                    Нормальный
                    Оскорбление
                    Угроза
                    Непристойность
                }
            }

            ... для каждого входного текста
        ]   
    }
"""
@app.route("/moderation/texts", methods=["POST"])
def moderation_texts():
    in_json = flask.request.json

    data = in_json["data"]

    if data is None:
        return json.dumps({"result": 404, "message": "Не указаны тексты для проверки."})

    try:
        result = {"result": 0, "data": moderation(data)}
    except Exception as exp:
        return json.dumps({"result": 404, "message": "Ошибка при обработке данных нейронной сетью.", "python_message": str(exp)})

    return json.dumps(result)



"""
    POST /color/texts
    json data:
    { data: ["...", "...", ...] }

    output:
    {
        result: 0 - успех, иначе ошибка
        message и python_message описание ошибки
        data: [
            {
                Значение: float
                Статус: -1 / 0 / 1
            },

            ... для каждого входного текста
        ]   
    }
"""
@app.route("/color/texts", methods=["POST"])
def color_texts():
    in_json = flask.request.json

    data = in_json["data"]

    if data is None:
        return json.dumps({"result": 404, "message": "Не указаны тексты для проверки."})

    try:
        result = {"result": 0, "data": color(data)}
    except Exception as exp:
        return json.dumps({"result": 404, "message": "Ошибка при обработке данных нейронной сетью.", "python_message": str(exp)})

    return json.dumps(result)



"""
    POST /company/texts
    json data:
    { data: ["...", "...", ...] }

    output:
    {
        result: 0 - успех, иначе ошибка
        message и python_message описание ошибки
        data: [
            {
                Код
                Название
            }

            ... для каждого входного текста
        ]   
    }
"""
@app.route("/company/texts", methods=["POST"])
def company_texts():
    in_json = flask.request.json

    data = in_json["data"]

    if data is None:
        return json.dumps({"result": 404, "message": "Не указаны тексты для проверки."})

    try:
        result = {"result": 0, "data": analytics.define_category(data, analytics.STD_CATEGORIES)}
    except Exception as exp:
        return json.dumps({"result": 404, "message": "Ошибка при обработке данных алгоритмом.", "python_message": str(exp)})

    return json.dumps(result)




"""
    POST /categories/texts
    json data:
    { data: ["...", "...", ...] }

    output:
    {
        result: 0 - успех, иначе ошибка
        message и python_message описание ошибки
        data: [
            [
                {
                    Код: 
                    Название: 
                },

                ....
            ],

            ...
        ]   
    }

    Модели нет, возвращает пустые списки категорий
"""
@app.route("/categories/texts", methods=["POST"])
def categories_texts():
    in_json = flask.request.json

    data = in_json["data"]

    if data is None:
        return json.dumps({"result": 404, "message": "Не указаны тексты для проверки."})

    try:
        result = {"result": 0, "data": [[] for _ in data]}
    except Exception as exp:
        return json.dumps({"result": 404, "message": "Ошибка при обработке данных нейронной сетью.", "python_message": str(exp)})

    return json.dumps(result)







if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)

