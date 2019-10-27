import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFINE_DATA_CSV = 1


def compare(X, w, y):
    '''
    :param X: 输入特征
    :param w: 权重
    :param y: 预测目标
    :return:
    '''
    scores = np.dot(X, w)
    y_pred = np.ones((scores[0], 1))
    loc_negtive = np.where(scores < 0)[0]
    y_pred[loc_negtive] = -1
    loc_wrong = np.where(y_pred != y)[0]

    return loc_wrong


def update(X, w, y):
    '''
    :param X: 输入特征
    :param w: 初始权重
    :param y: 目标
    :return:
    '''
    loc_wrong = compare(X, w, y)
    w = w + y[loc_wrong][0] * X[loc_wrong, :][0].reshape(3, 1)

    return w


def perceptron(X, w, y):
    '''
    :param X: 输入特征
    :param w: 权重
    :param y: 目标
    :return:
    '''
    iters, start = 0, time.time()

    while True:
        cnt, iters = len(compare(X, w, y)), iters + 1

        if (cnt <= 0): break
        print('迭代次数：{} 分类错误个数：{}'.format(iters, cnt))
        w = update(X, w, y)

    # end = time.time()  # 计算耗时，请注释print


if __name__ == '__main__':
    if DEFINE_DATA_CSV:
        data_txt = np.loadtxt('data.txt')
        data_df = pd.DataFrame(data_txt)
        data_df.to_csv('data_train.csv', index=False)

        data = pd.read_csv(r'data_train.csv', header=None)
        X, y = data.iloc[:, [0, 1]], data[2]
    elseif:


    X[2] = np.ones(X.shape[0], 1)  # 增加常数项
    X = X.values
