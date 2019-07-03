import os
import numpy as np
import pandas as pd
import src.models
import matplotlib.pyplot as plt
import collections
import pylab
from pylab import arange,pi,sin,cos,sqrt

fig_width_pt = 417  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'pdf',
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


#########################################
# Figures

# Target var distr
fig = pylab.figure()
for i, dataset in enumerate(datasets):
    # dataset = 'DOROTHEA'
    print("\n\n####################\n"+ dataset)
    data = src.load_data(dataset)
    ax = fig.add_subplot(2,3,i+1)
    ax.set_title(dataset)

    # make keys, vals and pos
    cnt = collections.Counter(data['y_train'])
    keys = [key for key, value in cnt.items()]
    values = [value for key, value in cnt.items()]
    pos = [x for x in range(len(cnt))]

    # attach to plot
    ax.bar(pos, values, color='gray')
    ax.set_xticks(pos)
    ax.set_xticklabels(keys)

pylab.tight_layout()
fig.savefig(os.path.join('tex', 'fig__distr_target_var.pdf'))


# input var distr
fig = pylab.figure()
for i, dataset in enumerate(datasets):
    # dataset = 'DOROTHEA'
    print("\n\n####################\n" + dataset)
    data = src.load_data(dataset)
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_title(dataset)

    if dataset in ['ARCENE', 'DEXTER', 'GISETTE', 'MADELON']:
        # min, max
        xmin = np.max(data['X_train'])
        xmax = np.min(data['X_train'])

        # make all bins
        bins_all = []
        rr = data['X_train'].shape[1] / 500
        for i2 in range(int(data['X_train'].shape[1]/rr)):
            vals, bins = np.histogram(data['X_train'][:,int(float(i2)*float(rr))])
            bins_adj = (bins[1] - bins[0])/2
            bins = bins + bins_adj
            bins = bins[:len(bins)-1]

            bins_all.append({'vals':vals, 'bins':bins})

            xmin = min(xmin, np.min(bins))
            xmax = max(xmax, np.max(bins))

        # set lim
        ax.set_xlim(left=xmin, right=xmax)

        # plot bins
        for bin in bins_all:
            ax.plot(bin['bins'], bin['vals'], color='gray', alpha=.1)

    elif dataset == 'DOROTHEA':

        cnts = []

        rr = data['X_train'].shape[1] / 500
        for i2 in range(int(data['X_train'].shape[1] / rr)):
            cnts.append(collections.Counter(data['X_train'][:,int(float(i2)*float(rr))]))

        ind = np.arange(len(cnts[0]))
        width = 0.65

        for i2, cnt in enumerate(cnts):
            ax.bar(ind - width / 2 + float(i2)*width/500, [val for key, val in cnt.items()], width/500, color='gray', alpha=0.95)

        ax.set_xticks(ind)
        ax.set_xticklabels(ind)


pylab.tight_layout()
fig.savefig(os.path.join('tex', 'fig__distr_input_var.pdf'))
pylab.show()

