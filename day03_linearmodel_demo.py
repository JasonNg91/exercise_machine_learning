#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/4/20 10:35 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day03_linearmodel_demo.py
# @Software: PyCharm

# 波士顿房价预测
# 代码1：线性回归-正规方程
# 代码2：线性回归-梯度下降
# 代码3：岭回归

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def linear1():
    """
    代码1：线性回归-正规方程
    :return:
    """
    # 1.加载数据集
    boston = load_boston()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3.标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5.得出模型
    print("正规方程-权重系数：\n", estimator.coef_)
    print("正规方程-偏置为：\n", estimator.intercept_)

    # 6.模型评估
    y_predit = estimator.predict(x_test)
    print("预测的房价为：\n", y_predit)
    error = mean_squared_error(y_test, y_predit)
    print("正规方程-均方误差为：\n", error)

    return None


def linear2():
    """
    代码2：线性回归-梯度下降
    :return:
    """
    # 1.加载数据集
    boston = load_boston()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3.标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.预估器
    estimator = SGDRegressor(learning_rate="constant", eta0=0.001, max_iter=100)
    estimator.fit(x_train, y_train)

    # 5.得出模型
    print("梯度下降-权重系数：\n", estimator.coef_)
    print("梯度下降-偏置为：\n", estimator.intercept_)

    # 6.模型评估
    y_predit = estimator.predict(x_test)
    print("预测的房价为：\n", y_predit)
    error = mean_squared_error(y_test, y_predit)
    print("梯度下降-均方误差为：\n", error)

    return None


def linear3():
    """
    代码2：岭回归
    :return:
    """
    # 1.加载数据集
    boston = load_boston()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3.标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.预估器
    estimator = Ridge(alpha=0.5, max_iter=100)
    estimator.fit(x_train, y_train)

    # 5.得出模型
    print("岭回归-权重系数：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)

    # 6.模型评估
    y_predit = estimator.predict(x_test)
    print("预测的房价为：\n", y_predit)
    error = mean_squared_error(y_test, y_predit)
    print("岭回归-均方误差为：\n", error)

    return None


if __name__ == "__main__":
    # 代码1：线性回归-正规方程
    linear1()
    # 代码2：线性回归-梯度下降
    linear2()
    # 代码3：岭回归-梯度下降
    linear3()