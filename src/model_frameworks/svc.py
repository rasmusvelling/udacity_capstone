import sklearn.svm as svm
import pandas as pd
import src

# dataset = 'ARCENE'
# data = src.load_data(dataset)

def svc_lin(data, params=None):

    # if params not set
    if params is None:
        params = {
            'C': 1
        }

    X = data['X_train']
    y = data['y_train']

    model = svm.LinearSVC(
        C=params['C'],
        max_iter=25000000
    )

    model.fit(X, y)

    return model


def svc_poly(data, params=None):

    # if params not set
    if params is None:
        params = {
            'degree': 2,
            'gamma': 1,
            'C': 1
        }

    X = data['X_train']
    y = data['y_train']

    model = svm.SVC(
        gamma=params['gamma'],
        C=params['C'],
        kernel='poly',
        degree=params['degree'])

    model.fit(X, y)

    return model


def svc_rbf(data, params=None):

    # if params not set
    if params is None:
        params = {
            'gamma': 1,
            'C': 1
        }

    X = data['X_train']
    y = data['y_train']

    model = svm.SVC(
        gamma=params['gamma'],
        C=params['C'],
        kernel='rbf')

    model.fit(X, y)

    return model
