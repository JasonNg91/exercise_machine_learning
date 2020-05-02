#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/1/20 10:25 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_KNN_Kirs_demo.py
# @Software: PyCharm

# 代码1：利用KNN算法对莺尾草进行分类

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_kirs():
    """
    利用KNN算法对莺尾草进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()
    # print(dir(iris))

    # 2.划分数据集
    xtrain, xtest, ytrain, ytest = train_test_split(iris["data"], iris["target"], train_size=0.8)

    # 3.特征工程：标准化
    transfer = StandardScaler()
    xtrain = transfer.fit_transform(xtrain)
    xtest = transfer.transform(xtest)  # 注意：使用训练集的fit结果进行transform

    # 4.KNN算法估计器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(xtrain, ytrain)

    # 5.模型评估
    # 1）方法1：直接对比真实值和预测值
    ypredict = estimator.predict(xtest)
    print("ypredict = \n", ypredict)
    print("直接对比真实值和预测值: \n", ytest == ypredict)

    # 2）方法2：计算准确率
    score = estimator.score(xtest, ytest)
    print("准确率为: \n", score)

    return None


def knn_kirs_gscv():
    """
    利用KNN算法对莺尾草进行分类,添加网格搜索和交叉验证
    :return:
    """
    # 1.获取数据
    iris = load_iris()
    # print(dir(iris))

    # 2.划分数据集
    xtrain, xtest, ytrain, ytest = train_test_split(iris["data"], iris["target"], train_size=0.8)

    # 3.特征工程：标准化
    transfer = StandardScaler()
    xtrain = transfer.fit_transform(xtrain)
    xtest = transfer.transform(xtest)  # 注意：使用训练集的fit结果进行transform

    # 4.KNN算法估计器
    estimator = KNeighborsClassifier()
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(xtrain, ytrain)

    # 5.模型评估
    # 1）方法1：直接对比真实值和预测值
    ypredict = estimator.predict(xtest)
    print("ypredict = \n", ypredict)
    print("直接对比真实值和预测值: \n", ytest == ypredict)

    # 2）方法2：计算准确率
    score = estimator.score(xtest, ytest)
    print("准确率为: \n", score)

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
    # 代码1：利用KNN算法对莺尾草进行分类
    # knn_kirs()

    # 代码2：利用KNN算法对莺尾草进行分类,加入网格搜索和交叉验证
    knn_kirs_gscv()