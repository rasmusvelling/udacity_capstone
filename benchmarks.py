import pandas as pd
import numpy as np

import sklearn.metrics
from sklearn import tree

import src
import src.models


datasets = [
    'ARCENE',
    'DEXTER',
    'DOROTHEA',
    'GISETTE',
    'MADELON'
]


for dataset in datasets:
    # load data
    data = src.load_data(dataset)

    BER = src.models.benchmark_all_neg1(data)
    print("\nAll negs")
    print(dataset)
    print(BER)

    BER = src.models.benchmark_all_pls1(data)
    print("\nAll plus")
    print(dataset)
    print(BER)

    BER = src.models.benchmark_pca_logistic(data)
    print("\nPCA Logisitc reg")
    print(dataset)
    print(BER)




#
#
#
#
#
#
# # Basic Decision Tree
# clf = tree.DecisionTreeClassifier().fit(Xpca_train, y_train)
# y_valid_hat = clf.predict(Xpca_valid)
# y_train_hat = clf.predict(Xpca_train)
# err_rate_train = 1-sklearn.metrics.accuracy_score(y_true=y_train, y_pred=y_train_hat)
# err_rate_valid = 1-sklearn.metrics.accuracy_score(y_true=y_valid, y_pred=y_valid_hat)
#
# 1-sklearn.metrics.accuracy_score(y_true=np.concatenate([y_train, y_valid]), y_pred=np.concatenate([y_train_hat, y_valid_hat]))
#
#
# print("BER: " + str((err_rate_train + err_rate_valid)/2))