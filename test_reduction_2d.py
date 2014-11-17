# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import sugarbee.reduction as reduction

if __name__ == '__main__':
    datasize = 100 
    df, headers = preprocessing.get_preprocessed_data(datasize)
#    print df["attack"].values.tolist()
    proj = reduction.reduction(df, n_components=2)
    print df.shape
    print proj.shape
    plt.scatter(proj[:,0], proj[:,1], c=df["attack"].values.tolist())
    plt.colorbar(ticks=range(0,20))

    plt.show()
