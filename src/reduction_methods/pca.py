from sklearn.decomposition import PCA

def pca(data, n_components=5):
    pca = PCA(n_components=n_components).fit(data['X_train'])
    Xred_train = pca.transform(data['X_train'])
    Xred_valid = pca.transform(data['X_valid'])

    return Xred_train, Xred_valid
