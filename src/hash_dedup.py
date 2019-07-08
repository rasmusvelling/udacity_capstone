import hashlib


def hash_and_deduplicate(param_grid):
    print("Hashing and de-duplicating. Estimated time :  " + str(round(param_grid.shape[0] / (27000/3), ndigits=1)) + "  minutes")

    param_grid = param_grid.copy().reset_index(drop=True)
    param_grid['hash_id'] = ''

    for idx, row in param_grid.iterrows():
        row_id = ""
        for key, value in row.iteritems():
            if key is not 'hash_id':
                row_id = row_id + str(key) + ":" + str(value) + "; "
        param_grid.loc[idx, 'hash_id'] = hashlib.sha224(row_id.encode('utf-8')).hexdigest()

    if len(param_grid['hash_id'].tolist()) != len(param_grid['hash_id'].unique().tolist()):
        print("removing duplicates")
        param_grid = param_grid.drop_duplicates(subset='hash_id', keep='first', inplace=False)
    else:
        print('no duplicates found')

    param_grid = param_grid.copy().reset_index(drop=True)

    return param_grid
