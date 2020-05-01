#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/1/20 10:25 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_machine_learning.py
# @Software: PyCharm

# 代码1：利用KNN算法对莺尾草进行分类

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


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


if __name__ == "__main__":
    # 代码1：利用KNN算法对莺尾草进行分类
    knn_kirs()
