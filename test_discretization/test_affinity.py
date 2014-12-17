# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import copy

from sklearn.cluster import k_means
from sklearn.manifold import spectral_embedding
from sklearn.utils import check_random_state

import nslkdd.preprocessing as preprocessing
import sugarbee.reduction as reduction
import sugarbee.distance as distance
import sugarbee.affinity as affinity

#def assign_undirected_weight(W, i, j, v):
#    W[i,j] = W[j,i] = v

if __name__ == '__main__':
    df, headers, gmms = preprocessing.get_preprocessed_data()
    df = df[0:10]
    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    proj = reduction.gmm_reduction(df_train, headers, gmms)

#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='euclidean', knn=3)
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='manhattan', knn=2)
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.gaussian, knn=5)
    A = affinity.get_affinity_matrix(proj, metric_method=distance.cosdist, knn=5)

    D = affinity.get_degree_matrix(A)
    
    print A
    print D

    W = A
    D[D == 0] = 1e-8 #don't laugh, there is a core package in R actually does this
    D_hat = D**(-0.5)
    L = D_hat * (D - W) * D_hat
    
    U, S, V = np.linalg.svd(L, full_matrices=True)
    target = np.argmax(np.absolute(np.diff(S)))
    print V.transpose()[:,target+1]
