from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def reducer_pca(data, params):

    if params is None:
        params = {
            'n_components': 5
        }

    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=params['n_components']))])
    pipeline.fit(data['X_train'])

    do = deepcopy(data)
    do['X_train'] = pipeline.transform(data['X_train'])
    do['X_valid'] = pipeline.transform(data['X_valid'])

    return do
