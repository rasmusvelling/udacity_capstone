import sklearn.svm as svm
# dataset = 'ARCENE'
# data = src.load_data(dataset)


def mod_svc_lin(data, params=None):

    # if params not set
    if params is None:
        params = {
            'C': 1
        }

    X = data['X_train']
    y = data['y_train']

    model = svm.LinearSVC(
        max_iter=9000000
    )

    model.fit(X, y)

    return model


def mod_svc_poly(data, params=None):

    # if params not set
    if params is None:
        params = {
            'degree': 2,
            'gamma': 1,
            'C': 1
        }

    X = data['X_train']
    y = data['y_train']

    if 'dataset' in params:
        pass

    model = svm.SVC(
        gamma='auto',
        C=params['C'],
        kernel='poly',
        degree=params['degree'])

    model.fit(X, y)

    return model


def mod_svc_rbf(data, params=None):

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
