import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, RepeatedKFold

DEFINE_NON_LINEARABLE = 1
# 0: 按比例划分 1：留一法 2：P次K折
DEFINE_DATA_SPLIT = 0

if __name__ == '__main__':
    if DEFINE_NON_LINEARABLE == 0:
        data = pd.read_csv('bill_authentication.csv')
    else:
        colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
        data = pd.read_csv('iris.data', names=colnames)

    X = data.drop('Class', axis=1)
    y = data['Class']

    if DEFINE_DATA_SPLIT == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    elif DEFINE_DATA_SPLIT == 1:
        loo = LeaveOneOut()
        train, test = loo.split(X)
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    elif DEFINE_DATA_SPLIT == 2:
        kf = KFold(n_splits=3)
        # kf = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        train, test = kf.split(X)
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    if DEFINE_NON_LINEARABLE == 0:
        svc_classifier = SVC(kernel='linear')
    else:
        svc_classifier = SVC(C=1.0, kernel='poly', degree=8)
        # svc_classifier = SVC(C=1.0, kernel='rbf')
        # svc_classifier = SVC(C=1.0, kernel='sigmoid')

    svc_classifier.fit(X_train, y_train)

    y_pred = svc_classifier.predict(X_test)

    print(classification_report(y_test, y_pred))
