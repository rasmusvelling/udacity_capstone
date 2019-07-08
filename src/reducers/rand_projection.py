from copy import deepcopy
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


def reducer_rand_proj_sparse(data, params):

    if params is None:
        params = {
            'n_components': 5
        }

    X = data['X_train']
    y = data['y_train']

    reducer = SparseRandomProjection(n_components=params['n_components'])
    reducer.fit(X)

    do = deepcopy(data)
    do['X_train'] = reducer.transform(data['X_train'])
    do['X_valid'] = reducer.transform(data['X_valid'])

    return do


def reducer_rand_proj_gauss(data, params):

    if params is None:
        params = {
            'n_components': 5
        }

    X = data['X_train']
    y = data['y_train']

    reducer = GaussianRandomProjection(n_components=params['n_components'])
    reducer.fit(X)

    do = deepcopy(data)
    do['X_train'] = reducer.transform(data['X_train'])
    do['X_valid'] = reducer.transform(data['X_valid'])

    return do