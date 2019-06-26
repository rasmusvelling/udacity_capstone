import pandas as pd


def ber(y, y_hat):
    # For dev
    # y = [1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1]
    # y_hat = [1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1]

    df = pd.DataFrame(data=y, columns=['y'])
    df['y_hat'] = y_hat

    a = df[(df['y']==-1) & (df['y_hat']==-1)].shape[0]
    b = df[(df['y']==-1) & (df['y_hat']==1)].shape[0]
    c = df[(df['y']==1) & (df['y_hat'] == -1)].shape[0]
    d = df[(df['y']==1) & (df['y_hat'] == 1)].shape[0]

    BER = 0.5 * (b / (a + b) + c / (c + d))

    return BER
