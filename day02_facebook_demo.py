#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/2/20 10:05 AM
# @Author  : mouyan.wu
# @Email   : mouyan.wu@gmail.com
# @File    : day02_facebook_demo.py
# @Software: PyCharm

# Facebook 查找入住位置练习
# 1.获取数据
# 2.数据处理：a.缩小数据范围 b.time->年月日十分秒 c.过滤签到次数少的地点 d.数据分割
# 3.特征工程：标准化
# 4.KNN算法估计器
# 5.模型选择和调优
# 6.模型评估

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knncls():
    """
    Facebook 查找入住位置练习
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("./train.csv")
    print(data.head(10))

    # 2.数据处理：
    # a.缩小数据范围
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")

    # b.time->年月日十分秒
    time_value = pd.to_datetime(data['time'], unit='s')
    print(time_value)
    # 时间转化为字典格式
    time_value = pd.DatetimeIndex(time_value)
    # 构造时间特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    # 删除time特征
    data = data.drop(['time'], axis=1)
    print(data)

    # c.过滤签到次数少的地点
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]
    # 取出特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    # d.数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 3.特征工程：标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 4.KNN算法估计器
    knn = KNeighborsClassifier()

    # 5.模型选择和调优:网格搜索和交叉验证
    param_dict = {"n_neighbors": [3, 5, 7, 10]}
    gc = GridSearchCV(knn, param_grid=param_dict, cv=3)
    gc.fit(x_train, y_train)

    # 6.模型评估
    # 计算准确率
    score = gc.score(x_test, y_test)
    print("准确率为: \n", score)

    # 最佳参数：best_params_
    print("最佳参数： \n", gc.best_params_)
    # 最佳结果：best_score_
    print("最佳结果： \n", gc.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器： \n", gc.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果： \n", gc.cv_results_)
    return None


if __name__ == "__main__":
    knncls()
