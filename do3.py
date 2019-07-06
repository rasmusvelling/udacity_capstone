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

print("Reduction")
num_cores = multiprocessing.cpu_count()
results1 = Parallel(n_jobs=num_cores)(delayed(src.do_reduce)(idx=idx, params=params) for idx, params in grid[0:20].iterrows())


#################################

print("fit and predict")
grid = grid.sort_index()
num_cores = multiprocessing.cpu_count()
results1 = Parallel(n_jobs=num_cores)(delayed(src.do_fit)(idx=idx, params=params) for idx, params in grid[0:20].iterrows())
results2 = Parallel(n_jobs=num_cores)(delayed(src.do_predict)(idx=idx, params=params) for idx, params in grid.iterrows())

