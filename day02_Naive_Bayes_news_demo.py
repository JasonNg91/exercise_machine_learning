#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/2/20 2:52 PM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_Naive_Bayes_news_demo.py
# @Software: PyCharm

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def naive_bayes_news_gscv():
    """
    利用朴素贝叶斯算法对新闻进行分类
    :return:
    """
    # 1.获取数据
    news = fetch_20newsgroups(subset="all")
    # print(dir(news))

    # 2.划分数据集
    xtrain, xtest, ytrain, ytest = train_test_split(news.data, news.target, train_size=0.8)

    # 3.特征工程：TF-IDF文本特征抽取
    transfer = TfidfVectorizer()
    xtrain = transfer.fit_transform(xtrain)
    xtest = transfer.transform(xtest)  # 注意：使用训练集的fit结果进行transform

    # 4.朴素贝叶斯算法
    estimator = MultinomialNB()
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
    naive_bayes_news_gscv()
