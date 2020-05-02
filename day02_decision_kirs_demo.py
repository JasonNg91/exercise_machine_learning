#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/2/20 4:12 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_decision_kirs_demo.py
# @Software: PyCharm

# 代码1：利用决策树算法对莺尾草进行分类

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision_kirs():
    """
    利用决策树算法对莺尾草进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()
    # print(dir(iris))

    # 2.划分数据集
    xtrain, xtest, ytrain, ytest = train_test_split(iris["data"], iris["target"], train_size=0.8)

    # 3.决策树算法估计器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(xtrain, ytrain)

    # 4.模型评估
    # 1）方法1：直接对比真实值和预测值
    ypredict = estimator.predict(xtest)
    print("ypredict = \n", ypredict)
    print("直接对比真实值和预测值: \n", ytest == ypredict)

    # 2）方法2：计算准确率
    score = estimator.score(xtest, ytest)
    print("准确率为: \n", score)

    # 5.决策树可视化(结合网站:www.webgraphviz.com)
    export_graphviz(estimator,out_file="iris_decision_tree.dot",feature_names=iris.feature_names)

    return None


if __name__ == "__main__":
    # 代码1：利用决策树算法对莺尾草进行分类
    decision_kirs()
