# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import copy

import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import sugarbee.reduction as reduction
import sugarbee.distance as distance

if __name__ == '__main__':
    datasize = 2 
    df, headers = preprocessing.get_preprocessed_data(datasize)
    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    proj = reduction.reduction(df_train, n_components=2)
    print proj
    print proj[0]
    print proj[1]
    dist = distance.gaussian(proj[0], proj[1])
    print dist
