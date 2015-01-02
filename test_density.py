# -*- coding: utf-8 -*-

import numpy as np
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from matplotlib import gridspec
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model
from nslkdd.get_kdd_dataframe import attack_types
from nslkdd.get_kdd_dataframe import df_by_attack_type
import sugarbee.reduction as reduction
import sugarbee.distance as distance
import sugarbee.affinity as affinity
import sugarbee.solver as solver

from autosp import predict_k
import colorhex
import util
import logger

def q(x, y):
    g = mlab.bivariate_normal(x, y, 4.38, 6.5, 15, -11)
    g0 = mlab.bivariate_normal(x, y, 4.9, 4.3, -15, 14)
    g1 = mlab.bivariate_normal(x, y, 2.8, 2.4, 16, -12)
    g2 = mlab.bivariate_normal(x, y, 4.6, 7.3, 14, -2)
    g3 = mlab.bivariate_normal(x, y, 3.43, 2.32, -15, 7)
    g4 = mlab.bivariate_normal(x, y, 6.3, 4.24, 13, -16)

    return g + (128/264.0)*g0 + (130/264.0)*g1 + (86/264.0)*g2 + (90/264.0)*g3 + (66/264.0)*g4
#    return 0.6*g1+28.4*g2/(0.6+28.4)

def test():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-20, 20, 0.1)
    Y = np.arange(-20, 20, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = q(X, Y)
    print Z.shape
    s = 0
    print Z[0,0]
    print Z[399,399]
    for x in range(400) :
        for y in range(400):
            s = s + Z[x,y]
    print s
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'),
                    linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)

#    plt.savefig('3dgauss.png')
#    plt.clf()
    
    plt.show()

if __name__ == '__main__':
    headers, attacks = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    title = "training20_only"
    logger.debug("#################################################")
    logger.debug(title)
    test()



