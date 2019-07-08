import lightgbm as lgbm
# dataset = 'ARCENE'
# data = src.load_data(dataset)


def mod_lightgbm(data, params=None):

    X = data['X_train']
    y = data['y_train']

    model = lgbm.LGBMClassifier()
    model.fit(X, y)

    return model
