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

    df, headers, gmms = preprocessing.get_preprocessed_data()
    df = df[0:1000]

    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    print "reductioning..."
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    cproj = copy.deepcopy(proj)
    print "plotting..."

    lists = []
    for i in range(22):
        lists.append([])
    attacks = df["attack"].values.tolist()

    for i, d in enumerate(cproj):
        lists[attacks[i]].append(d)

    plt.subplot(2, 1, 1)

    for i, p in enumerate(lists) :
        print "---"
        print p
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
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='manhattan', knn=30)
    A = affinity.get_affinity_matrix(proj, metric_method=distance.cosdist, metric_param='manhattan', knn=8)

    k = predict_k(A)
    if k > 5 :
        k = 5
    print A

    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    print k

    D = affinity.get_degree_matrix(A)
    L = affinity.get_laplacian_matrix(A,D)

    X = solver.solve(L)
    est = KMeans(n_clusters=k)
    est.fit(cproj)
    res = est.labels_
    res = sc.labels_
    print res

    lists = []
    for i in range(22):
        lists.append([])
    attacks = df["attack"].values.tolist()

    plt.subplot(2, 1, 2)
    for i, p in enumerate(cproj):
        plt.scatter(p[0], p[1], c=colormaps[res[i]])

    elapsed = (time.time() - start)

    plt.show()
