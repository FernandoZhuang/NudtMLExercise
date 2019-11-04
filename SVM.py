import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, RepeatedKFold

DEFINE_NON_LINEARABLE = 0
DEFINE_VISUALIZE = 1  # 是否开启可视化。如果开启可视化，在Irish数据集中，只能选择两种属性，如此方能画在二维平面上。
DEFINE_TWO_CLASS = 1
DEFINE_DATA_SPLIT = 0  # 0: 按比例划分 1：留一法 2：P次K折


def svm_multiple_class(X, y, classifier, resolution=0.02):
    '''
    :param X: 训练数据集
    :param y: 真实数据标签
    :param classifier: 训练完毕的SVM
    :param resolution: 图中的最小间距
    :return: void
    '''
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    plt.show()


def svm_two_class(X, y, classifier):
    '''
    :param X: 预测数据集
    :param y: 真实标签
    :param classifier: SVM
    :return: void
    '''
    ax = plt.gca()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, yy = np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, linewidth=1,
               facecolors='none',
               edgecolors='k')
    plt.show()


def main(X_train, X_test, y_train, y_test):
    # region 归一化
    sc = StandardScaler()
    sc.fit(X_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    # endregion

    # region 核函数选择
    if DEFINE_NON_LINEARABLE == 0:
        svc_classifier = SVC(kernel='linear', C=10)
    else:
        # svc_classifier = SVC(C=1.0, kernel='poly', degree=8)
        svc_classifier = SVC(C=1.0, kernel='rbf')
        # svc_classifier = SVC(C=1.0, kernel='sigmoid')
    # endregion

    svc_classifier.fit(X_train, y_train)
    # HACK: 是否根据可视化，把以下代码封装进svm_show函数中
    y_predict = svc_classifier.predict(X_test)
    print(classification_report(y_test, y_predict))
    accuracies.append(accuracy_score(y_pred=y_predict, y_true=y_test))

    if DEFINE_VISUALIZE:
        if DEFINE_TWO_CLASS:
            # svm_two_class(X_test, y_test, svc_classifier)
            svm_two_class(X_train, y_train, svc_classifier)  # 使用训练数据集画图，空心圈会和支持向量重叠
        else:
            svm_multiple_class(X_test, y_test, svc_classifier)


if __name__ == '__main__':
    accuracies = []

    # region 线性和非线性数据的读取
    if DEFINE_NON_LINEARABLE == 0:
        # data = pd.read_csv('bill_authentication.csv')
        iris = datasets.load_iris()
        X = iris.data[:, [2, 3]]
        X = X[0:100, :]  # 100~150代表virginica。剔除virginica后，数据是线性可分的。之所以剔除数据，是为了尝试作图
        y = iris.target[0:100]
    else:
        # region 本地数据
        # colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
        # data = pd.read_csv('iris.data', names=colnames)
        # endregion

        # iris = datasets.load_iris()

        X, y = make_circles(100, factor=.1, noise=.1)  # 使用圆圈表示非线性可分数据集

    # X = data.drop('Class', axis=1) if DEFINE_VISUALIZE == 0 else iris.data[:, [2, 3]]
    # y = data['Class'] if DEFINE_VISUALIZE == 0 else iris.target
    # endregion

    # region 数据划分
    if DEFINE_DATA_SPLIT == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        main(X_train, X_test, y_train, y_test)
    elif DEFINE_DATA_SPLIT == 1:
        loo = LeaveOneOut()
        for train, test in loo.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            main(X_train, X_test, y_train, y_test)
    elif DEFINE_DATA_SPLIT == 2:
        # kf = KFold(n_splits=3)
        kf = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            main(X_train, X_test, y_train, y_test)

        print('K折交叉验证 模型准确度：{}'.format(np.mean(accuracies)))
    # endregion
