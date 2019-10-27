import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt

DEFINE_DATA_CSV = 0
DEFINE_DATA_RANDOM = 0
DEFINE_DATA_SEPERABLE = 1
DEFINE_POCKET_ITERS = 1000


def compare(X, w, y):
    '''
    :param X: 输入特征
    :param w: 权重
    :param y: 预测目标
    :return: 预测错误的index
    '''
    scores = np.dot(X, w)
    y_pred = np.ones((scores.shape[0]))
    loc_negtive = np.where(scores < 0)[0]
    y_pred[loc_negtive] = -1
    loc_wrong = np.where(y_pred != y)[0]

    return loc_wrong


def update(X, w, y, loc_wrong):
    '''
    :param X: 输入特征
    :param w: 初始权重
    :param y: 目标
    :return: 更新权重
    '''
    w = w + y[loc_wrong][0] * X[loc_wrong, :][0].reshape(3, 1)

    return w


def update_pocket(X, w, y, loc_wrong):
    '''
    :param X: 输入特征
    :param w: 权重
    :param y: 目标
    :param loc_wrong: 预测错误的Index
    :return:
    '''
    num = len(loc_wrong)
    w = w + y[loc_wrong][np.random.choice(num)] * X[loc_wrong, :][np.random.choice(num)].reshape(3, 1)

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
        loc_wrong = compare(X, w, y)
        cnt, iters = len(loc_wrong), iters + 1

        if (cnt <= 0): break
        print('迭代次数：{} 分类错误个数：{}'.format(iters, cnt))
        w = update(X, w, y, loc_wrong)

    # end = time.time()  # 计算耗时，请注释print
    print('迭代次数：{} 参数W：{}'.format(iters, w))


def pocket(X, w, y):
    '''
    :param X: 输入特征
    :param w: 权重
    :param y: 目标
    :return:
    '''
    loc_wrong = compare(X, w, y)
    cnt, best_len, best_w = 0, len(loc_wrong), w

    for i in range(DEFINE_POCKET_ITERS):
        cnt += 1
        loc_wrong = compare(X, w, y)
        print('迭代次数{} 分类错误个数{}'.format(cnt, len(loc_wrong)))
        if (len(loc_wrong) <= 0): break

        w = update(X, w, y, loc_wrong)
        loc_wrong = compare(X, w, y)
        if len(loc_wrong) < best_len:
            best_len = len(loc_wrong)
            best_w = w

    print('参数W：{}'.format(best_w))


if __name__ == '__main__':
    if DEFINE_DATA_CSV:
        data_txt = np.loadtxt('data.txt')
        data_df = pd.DataFrame(data_txt)
        data_df.to_csv('data_train.csv', index=False)

        data = pd.read_csv(r'data_train.csv', header=None)
        X, y = data.iloc[:, [0, 1]], data[2]
    elif DEFINE_DATA_RANDOM:
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1,
                                   flip_y=-1)
    elif DEFINE_DATA_SEPERABLE:
        X, y = make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10))

    # mean = X.mean(axis=0)
    # sigma = X.std(axis=0)
    # X = (X - mean) / sigma
    X = np.concatenate((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)), axis=1)  # 增加常数项
    y[np.where(y == 0)] = -1
    w = X[0].copy()
    w[2] = 0
    w = w.reshape(3, 1)

    # PLA
    perceptron(X, w, y)

    # POCKET
    pocket(X, w, y)
