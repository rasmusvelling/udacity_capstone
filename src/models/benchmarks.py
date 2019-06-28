import src
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def benchmark_pca_logistic(data):
    # make PCA Logistic reg
    pca = PCA(n_components=5).fit(data['X_train'])
    Xpca_train = pca.transform(data['X_train'])
    Xpca_valid = pca.transform(data['X_valid'])

    # Basic Logistic regression
    clf = LogisticRegression(
        random_state=0, solver='sag', multi_class='ovr',
        max_iter=1000).fit(Xpca_train, data['y_train'])
    y_valid_hat = clf.predict(Xpca_valid)

    BER = src.ber(y=data['y_valid'].tolist(), y_hat=y_valid_hat.tolist())
    return BER


def benchmark_all_pls1(data):
    y_valid_hat = [1 for x in range(len(data['y_valid'].tolist()))]
    BER = src.ber(y=data['y_valid'].tolist(), y_hat=y_valid_hat)
    return BER


def benchmark_all_neg1(data):
    y_valid_hat = [-1 for x in range(len(data['y_valid'].tolist()))]
    BER = src.ber(y=data['y_valid'].tolist(), y_hat=y_valid_hat)
    return BER
