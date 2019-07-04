import src
from sklearn.model_selection import KFold
import numpy as np

def cross_validation_BER(X, y, model):

    ber_cv = []

    kf = KFold(n_splits=10, random_state=1986, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_test_hat = model.predict(X_test)
        ber_cv.append(src.ber(y=y_test.tolist(), y_hat=y_test_hat.tolist()))

    ber_cv = np.mean(ber_cv)

    return ber_cv
