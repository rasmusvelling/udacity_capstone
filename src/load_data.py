import numpy as np
import pandas as pd


def load_data(dataset):
    # For testing
    # dataset = 'DOROTHEA'

    ###################################
    # File Paths
    data_X_train = "data_NIPS2003\\" + dataset.upper() + "\\" + dataset.upper() + "\\" + dataset.lower() + "_train.data"
    data_X_valid = "data_NIPS2003\\" + dataset.upper() + "\\" + dataset.upper() + "\\" + dataset.lower() + "_valid.data"
    data_y_train = "data_NIPS2003\\" + dataset.upper() + "\\" + dataset.upper() + "\\" + dataset.lower() + "_train.labels"
    data_y_valid = "data_NIPS2003\\" + dataset.upper() + "\\" + dataset.lower() + "_valid.labels"

    ###################################
    # Load
    if dataset.upper() in ['ARCENE', 'GISETTE', 'MADELON']:
        data_X_train = pd.read_csv(data_X_train, delimiter=' ', header=None)
        data_X_valid = pd.read_csv(data_X_valid, delimiter=' ', header=None)
        data_y_train = pd.read_csv(data_y_train, delimiter=' ', header=None)
        data_y_valid = pd.read_csv(data_y_valid, delimiter=' ', header=None)

    elif dataset == 'DEXTER':
        data_X_train = _load_dexter(filepath=data_X_train, rows=300, cols=20000)
        data_X_valid = _load_dexter(filepath=data_X_valid, rows=300, cols=20000)
        data_y_train = pd.read_csv(data_y_train, delimiter=' ', header=None)
        data_y_valid = pd.read_csv(data_y_valid, delimiter=' ', header=None)

    elif dataset == 'DOROTHEA':
        data_X_train = _load_dorothea(filepath=data_X_train, rows=800, cols=100000)
        data_X_valid = _load_dorothea(filepath=data_X_valid, rows=350, cols=100000)
        data_y_train = pd.read_csv(data_y_train, delimiter=' ', header=None)
        data_y_valid = pd.read_csv(data_y_valid, delimiter=' ', header=None)

    ###################################
    # Basic transforms
    if dataset.upper() in ['ARCENE', 'GISETTE', 'MADELON']:
        data_X_train = data_X_train.drop(columns=data_X_train.shape[1]-1, axis=1)
        data_X_valid = data_X_valid.drop(columns=data_X_valid.shape[1]-1, axis=1)

        # to numpy everything
        data_X_train = data_X_train.to_numpy()
        data_X_valid = data_X_valid.to_numpy()

    data_y_train = data_y_train.to_numpy().flatten()
    data_y_valid = data_y_valid.to_numpy().flatten()

    ###################################
    # Drop zero columns
    len(np.where(~data_X_train.any(axis=0))[0])
    dx = np.delete(data_X_train, [1], axis=1)

    data = {
        'X_train': data_X_train,
        'X_valid': data_X_valid,
        'y_train': data_y_train,
        'y_valid': data_y_valid}

    return data


def _load_dexter(filepath, rows, cols):
    df = np.zeros((rows, cols))

    with open(filepath) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            line = line.split(" ")

            for item in line:
                # item = line[0]
                item = item.split(":")
                if len(item) == 2:
                    df[int(cnt), int(item[0])] = int(item[1])
                    # df[int(cnt), int(item[0])]
            line = fp.readline()
            cnt += 1
    pass

    return df


def _load_dorothea(filepath, rows, cols):
    df = np.zeros((rows, cols))

    # fp=open(filepath)
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            line = line.split(" ")
            for item in line:
                if item != '\n':
                    # print(item)
                    df[int(cnt), int(item)-1] = 1
            line = fp.readline()
            cnt += 1
    pass

    return df
