import datetime
import time
import os
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import pickle
import src
import src.reducers
import src.model_frameworks

test_mode = True

for path in [
    os.path.join('cache'),
    os.path.join('cache', 'data_reduced'),
    os.path.join('cache', 'data_reduced_time'),
    os.path.join('cache', 'models_fitted'),
    os.path.join('cache', 'models_fitted_time'),
    os.path.join('cache', 'predict_ber')
]:
    if not os.path.isdir(path) : os.mkdir(path)


grid_path = os.path.join('cache', 'grid.csv')
if not os.path.isfile(grid_path):
    grid = src.make_grid()
    grid.to_csv(grid_path, index=False)
else:
    grid = pd.read_csv(grid_path)

#################################
if test_mode:
    from sklearn.utils import shuffle
    grid = shuffle(grid)


#################################
print("\n\n\nReduction")
num_cores = multiprocessing.cpu_count()
results0 = Parallel(n_jobs=num_cores)(delayed(src.do_reduce)(idx=idx, params=params) for idx, params in grid.iterrows())

print("\n\n\nfit")
#grid = grid.sort_index()
num_cores = multiprocessing.cpu_count()
results1 = Parallel(n_jobs=num_cores)(delayed(src.do_fit)(idx=idx, params=params) for idx, params in grid.iterrows())

print("\n\n\nPredict")
results2 = Parallel(n_jobs=num_cores)(delayed(src.do_predict)(idx=idx, params=params) for idx, params in grid.iterrows())

##################################
grid = grid.sort_index()

#################################
reduce_times = os.listdir(os.path.join('cache', 'data_reduced_time'))
reduce_times = [pd.read_csv(os.path.join('cache', 'data_reduced_time', x), index_col=0) for x in reduce_times]
reduce_times = pd.concat(reduce_times)

grid = pd.merge(grid, reduce_times, how='left', left_on='data_hash_id', right_on='data_hash_id')


#################################
fit_times = os.listdir(os.path.join('cache', 'models_fitted_time'))
fit_times = [pd.read_csv(os.path.join('cache', 'models_fitted_time', x), index_col=0) for x in fit_times]
fit_times = pd.concat(fit_times)

grid = pd.merge(grid, fit_times, how='left', left_on='hash_id', right_on='hash_id')

grid.to_csv(os.path.join('cache', 'grid_2.csv'))
