from sklearn.linear_model import LogisticRegression
# dataset = 'ARCENE'
# data = src.load_data(dataset)


def mod_logisticreg(data, params=None):

    X = data['X_train']
    y = data['y_train']

    model = LogisticRegression(
        random_state=0, solver='sag', multi_class='ovr',
        max_iter=100000)
    model.fit(X, y)

    return model
