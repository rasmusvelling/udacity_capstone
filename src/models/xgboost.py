from xgboost import XGBClassifier
import src

def model_xgboost(data):
    model = XGBClassifier()
    model.fit(data['X_train'], data['y_train'])

    y_valid_hat = model.predict(data['X_valid'])
    BER = src.ber(y=data['y_valid'].tolist(), y_hat=y_valid_hat.tolist())
    return BER


