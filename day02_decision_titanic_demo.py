#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/2/20 5:04 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_decision_titanic_demo.py
# @Software: PyCharm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# 决策树对泰坦尼克号进行预测生死

def decision_titanic():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据
    titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 处理数据，找出特征值和目标值
    x = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    print(x)
    # 缺失处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)

    # 特征工程,特征-》类别-》one_hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))
    # 决策树估计器
    estimator = DecisionTreeClassifier()
    estimator = estimator.fit(x_train, y_train)
    # 预测准确率
    print("预测的准确率：", dict.score(x_test, y_test))

    # 导出决策树的结构
    export_graphviz(dict, out_file="./tree.dot",
                    feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    return None


if __name__ == "__main__":
    decision_titanic()
