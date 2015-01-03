# -*- coding: utf-8 -*-

import numpy as np
import copy

import pandas as pd
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

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

k = 12
num_clusters = k * 2
plot_lim_max = 21
plot_lim_min = -21

def check_abnormal_with_density(meanx, meany, stdx, stdy, target_sz):
    normal_sz = 264
    target_sz = target_sz
    X = np.arange(plot_lim_min, plot_lim_max, 1)
    Y = np.arange(plot_lim_min, plot_lim_max, 1)

    mX, mY = np.meshgrid(X, Y)
    n1 = mlab.bivariate_normal(mX, mY, 2.15, 0.89, 15.31, -6.5)
    n2 = mlab.bivariate_normal(mX, mY, 3.16, 3.21, 18, -17.5)
    n3 = mlab.bivariate_normal(mX, mY, 1.79, 1, 17.5, -11.3)
    n4 = mlab.bivariate_normal(mX, mY, 3.6, 2.5, 16, -11.4)
    n5 = mlab.bivariate_normal(mX, mY, 3.4, 2.6, 14, -18)
    n6 = mlab.bivariate_normal(mX, mY, 3.65, 4.85, 11, 4.7)
    n7 = mlab.bivariate_normal(mX, mY, 1.85, 1.45, 15.8, -13)
    normal_dist = n1 + n2 + n3 + n4 + n5 + n6 + n7

    target_dist = mlab.bivariate_normal(mX, mY, stdx, stdy, meanx, meany)
    print "===="
    print stdx
    print stdy
    print meanx
    print meany

    normal_dist = normal_dist*(normal_sz/float(normal_sz+target_sz))
    target_dist = target_dist*(target_sz/float(target_sz+target_sz))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(mX, mY, target_dist, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(mX, mY, normal_dist, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

#    print normal_dist
#    print target_dist
#    print normal_dist - target_dist

    s = 0
    for x in range(normal_dist.shape[0]) :
        for y in range(normal_dist.shape[1]):
            det = target_dist[x,y] - normal_dist[x,y]
            if det > 0:
                s = s + det
    return s

def test_clustering(df, gmms, title="", save_to_file=False, highlight_point=None):
    # preprocessing
    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    # from about 30 dimension to 2 dimension
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    cproj = copy.deepcopy(proj)

    # data_per_true_labels : try to make sort of dictionary per each label
    data_per_true_labels = []
    for i in range( len(attacks) ):
        data_per_true_labels.append([])

    true_attack_types = df["attack"].values.tolist()
    for i, d in enumerate(cproj):
        data_per_true_labels[true_attack_types[i]].append(d)

    A = affinity.get_affinity_matrix(cproj, metric_method=distance.cosdist, knn=8)

    k = predict_k(A)
    print "supposed k : " + str(k)

    lim = int(len(df) * 0.01)
    lim = 12
#    if lim < 3 or lim > 10 :
#        lim = 10
    k = lim
    print "Total number of clusters : " + str(k)

    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    res = sc.labels_

    # cluster data set
    clusters = [0] * k
    clusters_data = []
    clusters_xmean = [-1] * k
    clusters_ymean = [-1] * k
    clusters_xstd = [-1] * k
    clusters_ystd = [-1] * k
    for i in range(k) :
        clusters_data.append([])
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1
        clusters_data[ res[i] ].append(p)

    # cluster recheck with density
    for i, cluster in enumerate(clusters) :
        p = clusters_data[i]
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        clusters_xmean[i] = np.mean(x)
        clusters_ymean[i] = np.mean(y)
        clusters_xstd[i] = np.std(x)
        clusters_ystd[i] = np.std(y)

    ds = []
    for i, cluster in enumerate(clusters) :
        if cluster > 0 :
            d = check_abnormal_with_density(clusters_xmean[i],
                clusters_ymean[i],
                clusters_xstd[i],
                clusters_ystd[i],
                len(clusters_data[i]))
            ds.append(d)
            if 0 > d:
                clusters[i] = -99999
        else :
            ds.append(None)
    print ("ds")
    print ds

if __name__ == '__main__':
    """ Anomaly detection with spectral clustering algorithm.
    First training set only, to see what would happen with only known classes
    Next with test set, to see what would happen with only unknown classes
    """
    import time
    start = time.time()

    headers, attacks = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    """ here we get trainingset """
    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()
    gmms = gmms_20
    df1 = df_training_20[0:100]
    attack_type_index = 34;

    """ here we get testest """
    print "=== target ==="
    print model.attack_types[attack_type_index]
    df2 = df_by_attack_type(df_test_20, attack_type_index)
    df2 = df2[0:100]
    df = pd.concat([df1, df2])

    test_clustering(df, gmms, title="", save_to_file=False, highlight_point=attack_type_index)
    elapsed = (time.time() - start)
    print ("done in %s seconds" % (elapsed))
