import os
import pandas as pd
import pickle
import time
import datetime
import src
import src.reducers
import src.model_frameworks


def do_reduce(idx, params, test_mode=True):
    start = time.time()
    if idx % 100 == 0 or test_mode: print(str(idx) + "   " + str(datetime.datetime.now()))

    # reduce data
    data_path = os.path.join('cache', 'data_reduced', (params['data_hash_id'] + '.pkl'))
    if not os.path.isfile(data_path):
        print('applying data reduction')
        data = src.load_data(params['dataset'])

        # make dim reduction
        reducer = getattr(src.reducers, params['reducer'])
        data_reduced = reducer(data, params)

        # pickle reduced data
        pickle.dump(data_reduced, open(data_path, 'wb'))

        total_time = round(time.time() - start, ndigits=1)
        time_out = {
            'data_hash_id': params['data_hash_id'],
            'time_reduce': total_time
        }
        time_out = pd.DataFrame(time_out, index=[idx])
        time_out.to_csv(os.path.join('cache', 'data_reduced_time', (params['data_hash_id'] + '.csv')))


def do_fit(idx, params, test_mode=True):
    # fit models
    start = time.time()
    if idx % 100 == 0 or test_mode: print(str(idx) + "   " + str(datetime.datetime.now()))

    model_path = os.path.join('cache', 'models_fitted', (params['hash_id'] + '.pkl'))
    if not os.path.isfile(model_path):
        print(str(idx) + "  fitting model")

        # fit model
        data_path = os.path.join('cache', 'data_reduced', (params['data_hash_id'] + '.pkl'))
        model_framework = getattr(src.model_frameworks, params['model_framework'])
        data = pickle.load(open(data_path, 'rb'))
        model_fitted = model_framework(data, params)

        # pickle final model fit
        pickle.dump(model_fitted, open(model_path, 'wb'))

        total_time = round(time.time() - start, ndigits=1)
        time_out = {
            'data_hash_id': params['data_hash_id'],
            'time_reduce': total_time
        }
        time_out = pd.DataFrame(time_out, index=[idx])
        time_out.to_csv(os.path.join('cache', 'models_fitted_time', (params['data_hash_id'] + '.csv')))

        print(str(idx) + "  fit time:  " + str(total_time))


def do_predict(idx, params):
    # do prediction
    # load model
    model_path = os.path.join('cache', 'models_fitted', (params['hash_id'] + '.pkl'))
    model_fitted = pickle.load(open(model_path, 'rb'))

    # load data
    data_path = os.path.join('cache', 'data_reduced', (params['data_hash_id'] + '.pkl'))
    data = pickle.load(open(data_path, 'rb'))

    # predict validation set
    y_valid_hat = model_fitted.predict(data['X_valid'])

    # output BER
    ber = src.ber(y=data['y_valid'].tolist(), y_hat=y_valid_hat.tolist())
    predict_ber = {
        'hash_id': params['hash_id'],
        'BER': ber
    }
    predict_ber = pd.DataFrame(predict_ber, index=[idx])
    predict_ber.to_csv(os.path.join('cache', 'predict_ber', (params['hash_id'] + '.csv')))
    print("Predicted BER:  " + str(ber))

