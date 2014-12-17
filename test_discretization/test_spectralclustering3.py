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

from sklearn.cluster import KMeans
import pandas as pd

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model
from nslkdd.get_kdd_dataframe import attack_types
from nslkdd.get_kdd_dataframe import df_by_attack_type
import sugarbee.reduction as reduction
import sugarbee.distance as distance
import sugarbee.affinity as affinity
import sugarbee.solver as solver

from autosp import predict_k
from sklearn.cluster import SpectralClustering
import colorhex

plot_lim_max = 30
plot_lim_min = -30

def test_clustering(df, gmms, title="", save_to_file=False, point=None):
    df_train = copy.deepcopy(df)
    true_values = df_train["attack"].values.tolist()
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

#    print "reductioning..."
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    cproj = copy.deepcopy(proj)

#    print "plotting..."
    data_per_true_labels = []
    for i in range( len(attacks) ):
        data_per_true_labels.append([])
    true_attack_types = df["attack"].values.tolist()

    for i, d in enumerate(cproj):
        data_per_true_labels[true_attack_types[i]].append(d)

    fig, axarr = plt.subplots(3, 4, sharex='col', sharey='row')
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

    ax1.set_title("True labels")
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        colors = []
        if point == None :
            if i == model.attack_normal:
                colors.append('g')
            else :
                colors.append('r')
#            for _ in range(len(x)):
#                colors.append(colorhex.codes[i])
        else :
            for _ in range(len(x)):
                if i == point :
                    colors.append(colorhex.codes[i])
                elif i == model.attack_normal:
                    colors.append('g')
                else :
                    colors.append('r')

        ax1.scatter(x, y, c=colors)


##############################################################
    ax2.set_title("True normal")
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        if i == model.attack_normal:
            ax2.scatter(x, y, c='g')
##############################################################
    ax3.set_title("True abnormal")
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        if i != model.attack_normal:
            ax3.scatter(x, y, c='r')
##############################################################
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='euclidean', knn=8)
    A = affinity.get_affinity_matrix(proj, metric_method=distance.cosdist, knn=8)
#    D = affinity.get_degree_matrix(A)
#    L = affinity.get_laplacian_matrix(A,D)
#    X = solver.solve(L)
#    est = KMeans(n_clusters=k)
#    est.fit(cproj)
#    res = est.labels_

    k = predict_k(A)
    print "supposed k : " + str(k)

    lim = int(len(df) * 0.1)
    if k == 1 :
        k = lim
    if k > lim :
        k = lim
    print "Total number of clusters : " + str(k)

    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)

    res = sc.labels_
#    print "The results : "
#    print res
##############################################################
"""
what is this?
"""

    true_attack_types = df["attack"].values.tolist()

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
        if clusters[ res[i]] >= 0 :
            ax5.scatter(p[0], p[1], c='b')
##############################################################
    ax6.set_title("Abnormal res")
    for i, p in enumerate(cproj):
        if clusters[ res[i] ] < 0 :
            ax6.scatter(p[0], p[1], c='r')
##############################################################
    print res
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

    from sklearn.metrics import confusion_matrix
    m = confusion_matrix(list(y_true), list(y_pred))

    s1 = m[0][0] + m[0][1]
    s2 = m[1][1] + m[1][0]

    print m
    print "true_positive : " + str(m[0][0]) + " (" + str(m[0][0]*1.0 / s1) + ")"
    print "true_negative : " + str(m[1][1]) + " (" + str(m[1][1]*1.0 / s2) + ")"
    print "false_positive : " + str(m[1][0]) + " (" + str(m[1][0]*1.0 / s2) + ")"
    print "false_negative : " + str(m[0][1]) + " (" + str(m[0][1]*1.0 / s1) + ")"

    if save_to_file == True :
        fig.savefig("./plots/results/"+title+".png")
    else :
        plt.show()
    plt.close()

if __name__ == '__main__':
    import time
    start = time.time()

    headers, attacks = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    # with training-set
    df1 = df_training_20[0:500]
    gmms = gmms_20
    title = "training20_only"
    print "#################################################"
    print title
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
        print "#################################################"
        print title
        print len(df1)
        print len(df2)
        test_clustering(df, gmms, title=title, save_to_file=True, point=attack_type_index)

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)

