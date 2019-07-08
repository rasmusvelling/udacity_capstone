from xgboost import XGBClassifier
# dataset = 'ARCENE'
# data = src.load_data(dataset)


def mod_xgboost(data, params=None):

    X = data['X_train']
    y = data['y_train']

    model = XGBClassifier()
    model.fit(X, y)

    return model
