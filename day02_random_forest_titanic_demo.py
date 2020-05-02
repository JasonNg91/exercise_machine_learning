#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/2/20 5:04 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_random_forest_titanic_demo.py
# @Software: PyCharm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# 随机森林对泰坦尼克号进行预测生死

def random_forest_titanic():
    """
    随机森林对泰坦尼克号进行预测生死
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

    # 特征工程,特征->类别->one_hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 随机森林估计器
    estimator = RandomForestClassifier()
    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator = estimator.fit(x_train, y_train)
    # 预测准确率
    print("预测的准确率：", estimator.score(x_test, y_test))

    # 最佳参数：best_params_
    print("最佳参数： \n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果： \n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器： \n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果： \n", estimator.cv_results_)

    return None


if __name__ == "__main__":
    random_forest_titanic()
