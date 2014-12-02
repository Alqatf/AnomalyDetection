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
import pandas as pd
from sklearn.cluster import KMeans
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

today = util.make_today_folder('./results')
plot_lim_max = 30
plot_lim_min = -30

def plot_true_labels(ax, data_per_true_labels, title="", highlight_point = None):
    ax.set_title("True labels")
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        colors = []
        if highlight_point == None :
            if i == model.attack_normal:
                colors.append('g')
            else :
                colors.append('r')
#            for _ in range(len(x)):
#                colors.append(colorhex.codes[i])
        else :
            for _ in range(len(x)):
                if i == highlight_point :
                    colors.append(colorhex.codes[i])
                elif i == model.attack_normal:
                    colors.append('g')
                else :
                    colors.append('r')

        ax.scatter(x, y, c=colors)

def plot_normal_label(ax, data_per_true_labels, title=""):
    ax.set_title(title)
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        if i == model.attack_normal:
            ax.scatter(x, y, c='g')

def plot_abnormal_label(ax, data_per_true_labels, title=""):
    ax.set_title(title)
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        if i != model.attack_normal:
            ax.scatter(x, y, c='r')

def print_confusion_matrix(true_values, clusters, res):
    # confusion matrix
    y_true = []
    y_pred = []

    for i in true_values :
        if i == model.attack_normal :
            y_true.append(0)
        else :
            y_true.append(1)

    for i in res :
        if clusters[i] >= 0 :
            y_pred.append(0)
        else :
            y_pred.append(1)

    m = confusion_matrix(list(y_true), list(y_pred))

    s1 = m[0][0] + m[0][1]
    s2 = m[1][1] + m[1][0]

    logger.debug(m)
    logger.debug("true_positive : " + str(m[0][0]) + " (" + str(m[0][0]*1.0 / s1) + ")")
    logger.debug("true_negative : " + str(m[1][1]) + " (" + str(m[1][1]*1.0 / s2) + ")")
    logger.debug("false_positive : " + str(m[1][0]) + " (" + str(m[1][0]*1.0 / s2) + ")")
    logger.debug("false_negative : " + str(m[0][1]) + " (" + str(m[0][1]*1.0 / s1) + ")")

def test_clustering(df, gmms, title="", save_to_file=False, highlight_point=None):
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
#    A = affinity.get_affinity_matrix(cproj, metric_method=distance.dist, metric_param='euclidean', knn=8)

    k = predict_k(A)
    logger.debug("supposed k : " + str(k))

    lim = 10; #int(len(df) * 0.01)
    if k == 1 :
        k = lim
    if k > lim :
        k = lim
    logger.debug("Total number of clusters : " + str(k))

    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    res = sc.labels_
    logger.debug(res)

    # figure setting
    fig, axarr = plt.subplots(4, 4, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(plot_lim_min, plot_lim_max)
    plt.ylim(plot_lim_min, plot_lim_max)

    ax1 = axarr[0, 0]
    ax2 = axarr[0, 1]
    ax3 = axarr[0, 2]
    ax4 = axarr[0, 3]
    ax5 = axarr[1, 0]
    ax6 = axarr[1, 1]
    ax7 = axarr[1, 2]
    ax8 = axarr[1, 3]
    ax9 = axarr[2, 0]
    ax10 = axarr[2, 1]
    ax11 = axarr[2, 2]
    ax12 = axarr[2, 3]
    ax13 = axarr[3, 0]
    ax14 = axarr[3, 1]
    ax15 = axarr[3, 2]
    ax16 = axarr[3, 3]

    ##############################################################
    # plot true labels
    plot_true_labels(ax1, data_per_true_labels, "True labels")
    plot_normal_label(ax2, data_per_true_labels, "True normals")
    plot_abnormal_label(ax3, data_per_true_labels, "True abnormal")

    ##############################################################
    # plot predicted labels
    """
    As we already know training set labels, we can seperate normal data from "known" abnormal data and "unknown" abnormal data.
    """
    clusters = [0] * k
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1

    ##############################################################
    ax4.set_title("k-means")
    for i, p in enumerate(cproj):
        ax4.scatter(p[0], p[1], c=colorhex.codes[ res[i] ])

    ##############################################################
    ax5.set_title("Normal res")
    for i, p in enumerate(cproj):
        if clusters[ res[i] ] >= 0 :
            ax5.scatter(p[0], p[1], c='b')

    ##############################################################
    ax6.set_title("Abnormal res")
    for i, p in enumerate(cproj):
        if clusters[ res[i] ] < 0 :
            ax6.scatter(p[0], p[1], c='r')

    ##############################################################
    logger.debug("Cluster count")
    counts = [0] * k
    for _, c in enumerate(res):
        counts[c] = counts[c] + 1
    logger.debug(str(counts))

    ##############################################################
    ax7.set_title("Cluster 1")
    for i, p in enumerate(cproj):
        if res[i] == 0 :
            ax7.scatter(p[0], p[1], c='g')
    ##############################################################
    ax8.set_title("Cluster 2")
    for i, p in enumerate(cproj):
        if res[i] == 1 :
            ax8.scatter(p[0], p[1], c='g')
    ##############################################################
    ax9.set_title("Cluster 3")
    for i, p in enumerate(cproj):
        if res[i] == 2 :
            ax9.scatter(p[0], p[1], c='g')
    ##############################################################
    ax10.set_title("Cluster 4")
    for i, p in enumerate(cproj):
        if res[i] == 3 :
            ax10.scatter(p[0], p[1], c='g')
    ##############################################################
    ax11.set_title("Cluster 5")
    for i, p in enumerate(cproj):
        if res[i] == 4 :
            ax11.scatter(p[0], p[1], c='g')
    ##############################################################
    ax12.set_title("Cluster 6")
    for i, p in enumerate(cproj):
        if res[i] == 5 :
            ax12.scatter(p[0], p[1], c='g')
    ##############################################################
    ax13.set_title("Cluster 7")
    for i, p in enumerate(cproj):
        if res[i] == 6 :
            ax13.scatter(p[0], p[1], c='g')
    ##############################################################
    ax14.set_title("Cluster 8")
    for i, p in enumerate(cproj):
        if res[i] == 7 :
            ax14.scatter(p[0], p[1], c='g')
    ##############################################################
    ax15.set_title("Cluster 9")
    for i, p in enumerate(cproj):
        if res[i] == 8 :
            ax15.scatter(p[0], p[1], c='g')
    ##############################################################
    ax16.set_title("Cluster 10")
    for i, p in enumerate(cproj):
        if res[i] == 9 :
            ax16.scatter(p[0], p[1], c='g')
    ##############################################################

    print_confusion_matrix(true_attack_types, clusters, res)

    if save_to_file == True :
        fig.savefig(today + "/" + title + ".png")
    else :
        plt.show()
    plt.close()

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

    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    logger.set_file(today + "/log.txt")
    # with training-set
    df1 = df_training_20[0:500]
    gmms = gmms_20
    title = "training20_only"
    logger.debug("#################################################")
    logger.debug(title)
    test_clustering(df1, gmms, title=title, save_to_file=True)

    # with test-set
    dataset_description = "training20_test20"
    for attack_type_index, attack_type in enumerate(model.attack_types) :
        if attack_type_index <= model.attack_normal :
            continue
        df2 = df_by_attack_type(df_test_20, attack_type_index)
        df2 = df2[0:50]
        df = pd.concat([df1, df2])
        title = dataset_description + "_" + attack_type
        logger.debug("#################################################")
        logger.debug(title)
        logger.debug(str(len(df1)))
        logger.debug(str(len(df2)))
        test_clustering(df, gmms, title=title, save_to_file=True, highlight_point=attack_type_index)

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)
