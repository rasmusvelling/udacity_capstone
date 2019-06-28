import pandas as pd
import src.models

datasets = [
    'ARCENE',
    'DEXTER',
    'DOROTHEA',
    'GISETTE',
    'MADELON'
]

models = [
    'benchmark_all_neg1',
    'benchmark_all_pls1',
    'benchmark_pca_logistic',
    'model_xgboost'
]

rankings = []

for dataset in datasets:
    # load data
    # dataset = datasets[0]
    data = src.load_data(dataset)

    print("\n\n####################\n"+ dataset)

    for model in models:
        model_object = getattr(src.models, model)
        BER = model_object(data)
        rankings.append({
            'model': model,
            'dataset': dataset,
            'BER': BER
        })
        print(model)
        print(BER)

rankings = pd.concat()


#
#
#
#
#
#
# # Basic Decision Tree
# clf = tree.DecisionTreeClassifier().fit(Xpca_train, y_train)
# y_valid_hat = clf.predict(Xpca_valid)
# y_train_hat = clf.predict(Xpca_train)
# err_rate_train = 1-sklearn.metrics.accuracy_score(y_true=y_train, y_pred=y_train_hat)
# err_rate_valid = 1-sklearn.metrics.accuracy_score(y_true=y_valid, y_pred=y_valid_hat)
#
# 1-sklearn.metrics.accuracy_score(y_true=np.concatenate([y_train, y_valid]), y_pred=np.concatenate([y_train_hat, y_valid_hat]))
#
#
# print("BER: " + str((err_rate_train + err_rate_valid)/2))