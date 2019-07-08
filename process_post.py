import os
import numpy as np
import pandas as pd
import src.model_frameworks
import pylab

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


grid_path = os.path.join('cache', 'grid__inc_svc.csv')
if not os.path.isfile(grid_path):
    grid = src.make_grid()
    grid.to_csv(grid_path, index=False)
else:
    grid = pd.read_csv(grid_path)

grid = grid.sort_index()


#############################################3
print("\n\n\nPredict")
for idx, params in grid.iterrows():
    print(idx)
    src.do_predict(idx=idx, params=params)


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


#################################
predict_ber = os.listdir(os.path.join('cache', 'predict_ber'))
predict_ber = [pd.read_csv(os.path.join('cache', 'predict_ber', x), index_col=0) for x in predict_ber]
predict_ber = pd.concat(predict_ber)
grid = pd.merge(grid, predict_ber, how='left', left_on='hash_id', right_on='hash_id')


grid.to_csv(os.path.join('cache', 'grid_processed.csv'))


##################################
# Model performance

mp_incsvc = grid[grid['model_framework']=='mod_svc_lin'].copy()
mp_incsvc = mp_incsvc[~mp_incsvc['BER'].isnull()]
mp_incsvc = mp_incsvc['data_hash_id'].unique().tolist()
mp_incsvc = grid[grid['data_hash_id'].isin(mp_incsvc)].copy()
mp_incsvc.shape
mp_incsvc = mp_incsvc[['model_framework', 'BER']]
mp_incsvc = mp_incsvc.groupby('model_framework').mean()

mp_incsvc.to_latex()


mp = grid[grid['model_framework'] != 'mod_svc_lin'].copy()
mp = mp[['model_framework', 'BER']].copy()
mp = mp.groupby('model_framework').mean()


##################################
##################################
##################################
# data_reduce
data_reduce = grid.copy()[['reducer', 'data_hash_id', 'dataset', 'n_components', 'time_reduce']].drop_duplicates()
data_reduce['time_reduce'].sum()/60/60

# reduce times avg
reduce_times_avg = data_reduce[['reducer', 'time_reduce']].copy().groupby('reducer').agg([np.mean, np.median, np.std]).reset_index()
reduce_times_avg.to_latex()


##################################
# Reducer performance
reducer_performance = grid.copy()
reducer_performance = reducer_performance[reducer_performance['model_framework'] != 'mod_svc_lin']
reducer_performance_reducerwise = reducer_performance[['reducer', 'n_components', 'BER']].copy()
reducer_performance_reducerwise = reducer_performance_reducerwise.groupby(['reducer', 'n_components']).mean().reset_index()

reducer_performance = reducer_performance[['reducer', 'BER']].groupby('reducer').mean()
reducer_performance.to_latex()

# input var distr
fig = pylab.figure()
reducers = reducer_performance_reducerwise['reducer'].unique().tolist()
for i, reducer in enumerate(reducers):
    df = reducer_performance_reducerwise[reducer_performance_reducerwise['reducer']==reducer]

    ax = fig.add_subplot(2, 2, i + 1)
    ax.set_title(reducer)

    x=df['n_components'].unique().tolist()
    y=df['BER'].tolist()

    ax.set_ylim(bottom=0.1, top=0.5)

    ax.plot(x, y)

pylab.tight_layout()
fig.savefig(os.path.join('tex', 'fig__pred_perf_reduc_meth.pdf'))
pylab.show()
