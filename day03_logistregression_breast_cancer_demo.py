#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/5/20 8:43 AM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day03_logistregression_breast_cancer_demo.py
# @Software: PyCharm

# 代码：逻辑回归预测乳腺癌症
# 准确率/召回率
# ROC和AUC
# 模型保存和加载

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.externals import joblib


def logistregression_cancer():
    """
    # 代码：逻辑回归(二分类)预测乳腺癌症
    :return:
    """
    # 1.获取数据
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin" \
           ".data "
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
    data = pd.read_csv(path, names=column)

    # 2.缺失值数据处理
    data = data.replace(to_replace='?', value=np.nan)
    data.dropna(inplace=True)

    # 3.数据集划分
    x = data.iloc[:, 1:-1]
    y = data["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)

    # 4.特征工程
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 5.逻辑回归预估器
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 保存模型
    joblib.dump(estimator,"my_logistregression.pkl")

    # 模型加载
    # estimator = joblib.load("my_logistregression.pkl")

    # 6.得出模型
    print("回归系数\n", estimator.coef_)
    print("回归偏置\n", estimator.intercept_)

    # 7.模型评估
    # 1）方法1：直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict = \n", y_predict)
    print("直接对比真实值和预测值: \n", y_test == y_predict)

    # 2）方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: \n", score)

    # 3) 方法3：精确率和召回率
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
    print("精确率和召回率\n", report)

    # 4) 方法4：ROC和AUC
    y_true = np.where(y_test > 3, 1, 0)
    report = roc_auc_score(y_true, y_predict)
    print("ROC和AUC\n", report)

    return None


if __name__ == "__main__":
    logistregression_cancer()
