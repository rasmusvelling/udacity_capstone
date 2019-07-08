import os
import pandas as pd
import src.models

grid_path = os.path.join('cache', 'grid.csv')
if not os.path.isfile(grid_path):
    grid = src.make_grid()
    grid.to_csv(grid_path, index=False)
else:
    grid = pd.read_csv(grid_path)

# subset grid to pca 5 log reg
grid = grid[grid['reducer']=='reducer_pca']
grid = grid[grid['n_components']==5]
grid = grid[grid['model_framework']=='mod_logisticreg']


################################
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

wins = pd.DataFrame(src.models.benchmarks.benchmark_challenge_winner())
benchmark = grid.copy()[['BER', 'model_framework', 'dataset']].rename(columns={'model_framework':'model'})
benchmark = pd.concat([benchmark, wins], sort=True)

benchmark = pd.pivot_table(benchmark, values='BER', index=['model'], columns=['dataset'])
benchmark['Total'] = benchmark.mean(axis=1)

benchmark.to_latex()