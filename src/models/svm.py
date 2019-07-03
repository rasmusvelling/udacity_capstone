import sklearn.svm as svm
import src
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, mutual_info_classif


def model_svm(data):

    X = data['X_train']
    y = data['y_train']

    print(X.shape)
    X_new = SelectPercentile(mutual_info_classif, percentile=10).fit_transform(X, y)
    print(X_new.shape)

    c = [2**float(ci) for ci in range(-5,11,2)]
    g = [2**float(gi) for gi in range(-11,9,2)]
    degree = [d for d in range(0,6)]
    kernel = ['poly'] #, 'sigmoid', 'rbf']

    params = {
        'c': c,
        'g': g,
        'degree': degree,
        'kernel': kernel
    }

    grid = src.expand_grid(**params)
    grid = pd.DataFrame(grid)
    grid['BERcv'] = 1.0

    grid.shape
    if False:
        from sklearn.utils import shuffle
        grid = shuffle(grid)

    for idx, row in grid[0:20].iterrows():
        model = svm.SVC(gamma=row['g'], C=row['c'], kernel=row['kernel'], degree=row['degree'])

        BERs = []

        kf = KFold(n_splits=10, random_state=1986, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_test_hat = model.predict(X_test)
            BERs.append(src.ber(y=y_test.tolist(), y_hat=y_test_hat.tolist()))

        grid.loc[idx, 'BERcv'] = np.mean(BERs)


    best = grid[grid['BERcv'] == grid['BERcv'].min()]

    model = svm.SVC(gamma=np.median(best['g']), C=np.median(best['c']), kernel=row['kernel'])





    return BER


