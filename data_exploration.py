import os
import pandas as pd
import src.models
import matplotlib.pyplot as plt
import collections
import pylab
from pylab import arange,pi,sin,cos,sqrt

fig_width_pt = 418.25368  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)

datasets = [
    'ARCENE',
    'DEXTER',
    'DOROTHEA',
    'GISETTE',
    'MADELON'
]


fig, axs = pylab.subplots(nrows=2, ncols=3)
for i, dataset in enumerate(datasets):
    print("\n\n####################\n"+ dataset)
    print(str(i) + "  " + dataset)
    data = src.load_data(dataset)
    ax = fig.axes[i]
    ax.set_title(dataset)

    cnt = collections.Counter(data['y_train'])
    keys = [str(key) for key, value in cnt.items()]
    values = [value for key, value in cnt.items()]
    x = [x for x in range(len(cnt))]

    ax.bar(x, values)
    ax.set_xticks(keys)
    #print(dir(ax))


pylab.tight_layout()
fig.savefig(os.path.join('tex', 'fig__fig1.eps'))
fig.show()

#
#
# # Basic Decision Tree
# clf = tree.DecisionTreeClassifier().fit(Xpca_train, y_train)
# y_valid_hat = clf.predict(Xpca_valid)
# y_train_hat = clf.predict(Xpca_train)
# err_rate_train = 1-sklearn.metrics.accuracy_score(y_true=y_train, y_pred=y_train_hat)
# err_rate_valid = 1-sklearn.metrics.accuracy_score(y_true=y_valid, y_pred=y_valid_hat)
#
# 1-sklearn.metrics.accuracy_score(y_true=np.concatenate([y_train, y_valid]), y_pred=np.concatenate([y_train_hat, y_valid_hat]))
#
#
# print("BER: " + str((err_rate_train + err_rate_valid)/2))