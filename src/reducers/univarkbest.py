from copy import deepcopy
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def reducer_univarkbest(data, params):

    if params is None:
        params = {
            'n_components': 5
        }

    X = data['X_train']
    y = data['y_train']

    reducer = SelectKBest(mutual_info_classif, k=params['n_components'])
    reducer.fit(X, y)

    do = deepcopy(data)
    do['X_train'] = reducer.transform(data['X_train'])
    do['X_valid'] = reducer.transform(data['X_valid'])

    return do
