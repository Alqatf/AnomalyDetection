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
import sugarbee.affinity as affinity

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import *
from scipy import *
from autosp import predict_k

if __name__ == '__main__':
    attack_names = ("back","buffer_overflow","ftp_write","guess_passwd","imap",
    "ipsweep","land","loadmodule","multihop","neptune",
    "nmap","normal","perl","phf","pod",
    "portsweep","rootkit","satan","smurf","spy",
    "teardrop","warezclient","warezmaster")

    colormaps = ["b","g","r","c","m","k","w","0.20","0.75","#eeefff",
    "#000fff","#235234","#345454","#5766723","#263543","#078787","#567576","#745655","#958673","#262434",
    "#dd2453","#eee253","#fff332"]

    import time
    start = time.time()

    df, headers, gmms = preprocessing.get_preprocessed_data()
    df = df[0:100]

    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    print "reductioning..."
    proj = reduction.gmm_reduction(df_train, headers, gmms)

    A = affinity.get_affinity_matrix(proj, metric_method=distance.cosdist, knn=5)
    D = affinity.get_degree_matrix(A)

    print A

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)

    plt.show()
