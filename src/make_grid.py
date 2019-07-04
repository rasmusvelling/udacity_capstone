import pandas as pd
import src

def make_grid():
    #########################################

    def tidy_dimred(dimred_tmp):
        dimred_tmp = pd.DataFrame(src.expand_grid(**dimred_tmp))
        dimred_tmp = src.hash_and_deduplicate(dimred_tmp)
        dimred_tmp = dimred_tmp.rename(columns={'hash_id': 'data_hash_id'})
        return dimred_tmp

    #########################################
    # datasets
    datasets = ['ARCENE', 'DEXTER', 'DOROTHEA', 'GISETTE', 'MADELON']

    #########################################
    # make dim reduction methods param set
    dimred = []

    # PCA
    dimred_tmp = {
        'dataset': datasets,
        'reducer': ['reducer_pca'],
        'n_components': [5, 10, 25, 50]
    }
    dimred.append(tidy_dimred(dimred_tmp))

    # Univariate
    dimred_tmp = {
        'dataset': datasets,
        'reducer': ['reducer_univarkbest'],
        'n_components': [5, 10, 25, 50]
    }
    dimred.append(tidy_dimred(dimred_tmp))

    # Random Projections
    dimred_tmp = {
        'dataset': datasets,
        'reducer': ['reducer_rand_proj_sparse', 'reducer_rand_proj_gauss'],
        'n_components': [5, 10, 25, 50]
    }
    dimred.append(tidy_dimred(dimred_tmp))

    # finish up
    dimred = pd.concat(dimred, sort=True)
    dimred = dimred[['reducer'] + [col for col in dimred.columns.tolist() if col != 'reducer']]
    del dimred_tmp

    #########################################
    # make all model grids, join with dimred
    models = []

    # SVC line
    models_tmp = {
        'model_framework': ['svc_lin'],
        'C': [0.25, 1, 2.5]
    }
    models.append(pd.DataFrame(src.expand_grid(**models_tmp)))

    # # SVC poly
    # models_tmp = {
    #     'model_framework': ['svc_poly'],
    #     'gamma': [0.25, 1, 2.5],
    #     'C': [0.25, 1, 2.5],
    #     'degree': [1,3,5]
    # }
    # models.append(pd.DataFrame(src.expand_grid(**models_tmp)))
    #
    # # SVC rbf
    # models_tmp = {
    #     'model_framework': ['svc_rbf'],
    #     'gamma': [0.25, 1, 2.5],
    #     'C': [0.25, 1, 2.5]
    # }
    models.append(pd.DataFrame(src.expand_grid(**models_tmp)))

    models = pd.concat(models, sort=True)
    models = models[['model_framework'] + [col for col in models.columns.tolist() if col != 'model_framework']]
    del models_tmp

    #########################################
    # final join all
    models['key'] = 1
    dimred['key'] = 1
    grid = pd.merge(models, dimred, on='key').drop('key', axis=1)

    print("rows in grid == datasets * models * dimred: " + str(models.shape[0] * dimred.shape[0] == grid.shape[0]))
    grid = src.hash_and_deduplicate(grid)
    print('Rows in grid:  ' + str(grid.shape[0]) + ".  Len unique hash_id: " + str(
        len(grid['hash_id'].unique().tolist())))

    grid['time_reduce'] = -1
    grid['time_fit'] = -1
    grid['BER'] = 1

    return grid
