import pandas as pd


def load_data(dataset):
    # dataset = 'MADELON'

    # Paths
    data_X_train = dataset.upper() + "\\" + dataset.upper() + "\\" dataset.lower() + "_train.data"
    data_X_valid = dataset.upper() + "\\" + dataset.upper() + "\\" dataset.lower() + "_valid.data"
    data_y_train = dataset.upper() + "\\" + dataset.upper() + "\\" dataset.lower() + "_train.labels"
    data_y_valid = dataset.upper() + "\\" dataset.lower() + "_valid.labels"

    # Load
    data_X_train = pd.read_csv(data_X_train, delimiter=' ', header=None)
    data_X_valid = pd.read_csv(data_X_valid, delimiter=' ', header=None)
    data_y_train = pd.read_csv(data_y_train, delimiter=' ', header=None)
    data_y_valid = pd.read_csv(data_y_valid, delimiter=' ', header=None)

    # Basic transforms
    data_X_train = data_X_train.drop(columns=500, axis=1)
    data_X_valid = data_X_valid.drop(columns=500, axis=1)

    # to numpy everything
    data_X_train = data_X_train.to_numpy()
    data_X_valid = data_X_valid.to_numpy()
    data_y_train = data_y_train.to_numpy().flatten()
    data_y_valid = data_y_valid.to_numpy().flatten()

    return data_X_train, data_X_valid, data_y_train, data_y_valid


