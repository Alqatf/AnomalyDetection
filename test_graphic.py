# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import copy
import cPickle as pickle

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model
from nslkdd.get_kdd_dataframe import attack_types
from nslkdd.get_kdd_dataframe import df_by_attack_type

import colorhex
import util

plot_lim_max = 21
plot_lim_min = -21

if __name__ == '__main__':
    fig, ax= plt.subplots(1, 1, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(plot_lim_min, plot_lim_max)
    plt.ylim(plot_lim_min, plot_lim_max)
    plt.xlabel('interval')
    plt.ylabel('log(probability) + k')
    plt.title('Convergence plot')
    plt.grid(True)

    x = range(10)
    y = range(10)
    ax.scatter(x, y, c='r')
    plt.show()
