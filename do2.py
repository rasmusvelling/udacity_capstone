import datetime
import time
import os
import pandas as pd
import pickle
import src
import src.reducers
import src.model_frameworks

test_mode = True

if not os.path.isdir(os.path.join('cache')) : os.mkdir(os.path.join('cache'))
if not os.path.isdir(os.path.join('cache', 'data_reduced')) : os.mkdir(os.path.join('cache', 'data_reduced'))
if not os.path.isdir(os.path.join('cache', 'models_fitted')) : os.mkdir(os.path.join('cache', 'models_fitted'))

grid_path = os.path.join('cache', 'grid.csv')
if not os.path.isfile(grid_path):
    grid = src.make_grid()
    grid.to_csv(grid_path, index=False)
else:
    grid = pd.read_csv(grid_path)

# if test_mode:
#     from sklearn.utils import shuffle
#     grid = shuffle(grid)

#########################################
# do grid
chunck = 15
steps = grid.shape[0] // chunck + 1
iters = [x for x in range(int(steps))]

for iter in iters:
    # iter=26
    i = iter*chunck
    j = min(i + chunck, grid.shape[0])

    # reduce data
    for idx, params in grid[i:j].iterrows():
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

            total_time = round(time.time() - start, ndigits=0)
            grid.loc[idx, 'time_reduce'] = total_time
            grid.sort_index().to_csv(grid_path, index=False)

            # clean up
            del data, data_reduced, total_time

        elif test_mode:
            print('data already reduced')
    del idx, params, data_path, start


    # fit models
    for idx, params in grid[i:j].iterrows():
        start = time.time()
        if idx % 100 == 0 or test_mode: print(str(idx) + "   " + str(datetime.datetime.now()))

        model_path = os.path.join('cache', 'models_fitted', (params['hash_id'] + '.pkl'))
        if not os.path.isfile(model_path):
            print("fitting model")

            # fit model
            data_path = os.path.join('cache', 'data_reduced', (params['data_hash_id'] + '.pkl'))
            model_framework = getattr(src.model_frameworks, params['model_framework'])
            data = pickle.load(open(data_path, 'rb'))
            model_fitted = model_framework(data, params)

            # pickle final model fit
            pickle.dump(model_fitted, open(model_path, 'wb'))

            total_time = round(time.time() - start, ndigits=0)
            grid.loc[idx, 'time_fit'] = total_time
            grid.sort_index().to_csv(grid_path, index=False)

            # clean up
            del data_path, model_framework, data, model_fitted, total_time

    del idx, params, model_path, start


    # do prediction
    for idx, params in grid[i:j].iterrows():
        if idx % 100 == 0 or test_mode: print(str(idx) + "   " + str(datetime.datetime.now()))

        # load model
        model_path = os.path.join('cache', 'models_fitted', (params['hash_id'] + '.pkl'))
        model_fitted = pickle.load(open(model_path, 'rb'))

        # load data
        data_path = os.path.join('cache', 'data_reduced', (params['data_hash_id'] + '.pkl'))
        data = pickle.load(open(data_path, 'rb'))

        # predict validation set
        y_valid_hat = model_fitted.predict(data['X_valid'])

        # output BER
        grid.loc[idx, 'BER'] = src.ber(y=data['y_valid'].tolist(), y_hat=y_valid_hat.tolist())
        grid.sort_index().to_csv(grid_path, index=False)

    del idx, params, data, data_path, model_path, model_fitted, y_valid_hat

