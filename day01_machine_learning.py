#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/1/20 10:25 AM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day01_machine_learning.py
# @Software: PyCharm

# 特征提取
# 1.导入包
# 2.实例化
# 3.转换数据
# 4.打印结果

# 代码1:字典抽取
# 代码2：文本特征提取
# 代码3：中文特征值化
# 代码4：TF-IDF中文特征值化
# 代码5：归一化处理,容易受粗差野值影响
# 代码6：标准化处理
# 代码7：缺失值处理
# 代码8：过滤式特征选择
# 代码9：主成分分析特征降维

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def dictvec():
    """
    字典数据抽取
    :return:
    """
    # 实例化
    dict = DictVectorizer(sparse=False)
    # 调用fit_transform
    data = dict.fit_transform(
        [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}])
    print(dict.get_feature_names())
    # print(dict.inverse_transform(data))
    print(data)
    return None


def countvec():
    """
    对文本进行特征值化
    :return:
    """
    # 实例化
    cv = CountVectorizer()
    # 调用fit_transform
    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])
    # 打印输出
    print(cv.get_feature_names())
    print(data.toarray())
    return None


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    # 转换成list
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # list 转化字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1, c2, c3


def chinese_vec():
    """
    中文特征值化
    :return:
    """
    # 数据转化为字符串格式
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    # 实例化
    cv = CountVectorizer()
    # 调用fit_transform
    data = cv.fit_transform([c1, c2, c3])
    # 打印输出
    print(cv.get_feature_names())
    print(data.toarray)
    return None


def tfidfvec():
    """
    TF-IDF中文特征值化
    :return: 
    """
    # 数据转化为字符串格式
    c1, c2, c3 = cutword()
    # 实例化
    tf = TfidfVectorizer()
    # 调用fit_transform
    data = tf.fit_transform([c1, c2, c3])
    # 打印输出
    print(tf.get_feature_names())
    print(data.toarray())
    return None


def mm():
    """
    归一化缩放
    :return:
    """
    # 实例化
    mm = MinMaxScaler(feature_range=(0, 1))
    # 调用fit_transform
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    # 打印输出
    print(data)
    return None


def stand():
    """
    标准化缩放
    :return: 
    """
    # 实例化
    std = StandardScaler()
    # 调用fit_transform
    data = std.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    # 打印输出
    print(data)
    return None


def im():
    """
    缺失值处理
    :return:
    """
    # 实例化NaN, nan
    im = SimpleImputer(missing_values=np.nan, strategy="mean")
    # 调用fit_transform
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print('original data = \n', [[1, 2], [np.nan, 3], [7, 6]])
    # 打印输出
    print('new data = \n', data)
    return None


def var():
    """
    特征选择-过滤式删除低方差特征
    :return:
    """
    # 实例化
    var = VarianceThreshold(threshold=2.0)
    # 调用fit_transform
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    # 打印输出
    print(data)
    return None


def pca():
    """
    PCA主成分分析特征降维
    :return:
    """
    # 实例化
    pca = PCA(n_components=0.95)
    # 调用fit_transform
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    # 打印输出
    print(data)
    return None


if __name__ == "__main__":
    # 代码1:字典抽取
    # dictvec()

    # 代码2：文本特征提取
    # countvec()

    # 代码3：中文特征值化
    # chinese_vec()

    # 代码4：TF-IDF中文特征值化
    # tfidfvec()

    # 代码5：归一化处理,容易受粗差野值影响
    # mm()

    # 代码6：标准化处理
    # stand()

    # 代码7：缺失值处理
    # im()

    # 代码8：过滤式特征选择
    # var()

    # 代码9：主成分分析特征降维
    pca()
