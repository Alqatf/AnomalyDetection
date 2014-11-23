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

import nslkdd.preprocessing as preprocessing
import nslkdd.preprocessing as model
import sugarbee.reduction as reduction
import sugarbee.distance as distance
import sugarbee.affinity as affinity
import sugarbee.solver as solver

from autosp import predict_k
from sklearn.cluster import SpectralClustering

if __name__ == '__main__':
    attack_names = ("back","buffer_overflow","ftp_write","guess_passwd","imap",
    "ipsweep","land","loadmodule","multihop","neptune",
    "nmap","normal","perl","phf","pod",
    "portsweep","rootkit","satan","smurf","spy",
    "teardrop","warezclient","warezmaster")

    colormaps = ["b","r","m","c","k","0.1","w","0.20","0.75","#eeefff",
    "#000fff","#235234","#345454","#5766723","#263543","#078787","#567576","#745655","#958673","#262434",
    "#dd2453","#eee253","#fff332"]

    import time
    start = time.time()

    headers, attacks = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    df = df_training_20[0:200]
    gmms = gmms_20

    df_train = copy.deepcopy(df)
    true_values = df_train["attack"].values.to_list()
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    print "reductioning..."
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    cproj = copy.deepcopy(proj)
    print "plotting..."

    lists = []
    for i in range(22):
        lists.append([])
#    attacks = df["attack"].values.tolist()

    for i, d in enumerate(cproj):
        lists[attacks[i]].append(d)

    plt.axis([0,100.0,0,100.0])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.subplot(4, 1, 1)
    plt.title("True labels")

    for i, p in enumerate(lists) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        colors = []
        for _ in range(len(x)):
            colors.append(colormaps[i])
        plt.scatter(x, y, c=colors)

#    plt.legend(attack_names, loc='best')
    elapsed = (time.time() - start)

    print "done in %s seconds" % (elapsed)

    print "=============="
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='euclidean', knn=8)
    A = affinity.get_affinity_matrix(proj, metric_method=distance.cosdist, metric_param='manhattan', knn=8)

    k = predict_k(A)
    lim = int(len(df) * 0.1)
    if k > lim :
        print "supposed k : " + str(k)
        k = lim
    print "Total number of clusters : " + str(k)

    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)

    D = affinity.get_degree_matrix(A)
    L = affinity.get_laplacian_matrix(A,D)

    X = solver.solve(L)
#    est = KMeans(n_clusters=k)
#    est.fit(cproj)
#    res = est.labels_
    res = sc.labels_
    print "The results : "
    print res

    lists = []
    for i in range( len(attacks) ):
        lists.append([])
#    attacks = df["attack"].values.tolist()

    plt.subplot(4, 1, 2)
    plt.title("Spectral clustered")

##    cm = plt.cm.get_cmap('RdYlBu')
#    for i, p in enumerate(cproj):
#        plt.scatter(p[0], p[1], c=colormaps[res[i]])
##        plt.scatter(p[0], p[1], vmin=0, vmax=k, s=35, cmap=cm) #=colormaps[res[i]])

    plt.subplot(4, 1, 3) # normal
    plt.title("Normal clustered")

    clusters = [0] * k
    for i, p in enumerate(cproj):
        true_label = attacks[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1

    for i, p in enumerate(cproj):
        if clusters[ res[i]] >= 0 :
            plt.scatter(p[0], p[1], c='b')

    plt.subplot(4, 1, 4) # abnormal
    plt.title("Abnormal clustered")

    for i, p in enumerate(cproj):
        if clusters[ res[i] ] < 0 :
            plt.scatter(p[0], p[1], c='r')


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

    elapsed = (time.time() - start)
    plt.show()
