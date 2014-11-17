# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np

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
    datasize = 5 
    df, headers = preprocessing.get_preprocessed_data(datasize)
    proj = reduction.reduction(df, n_components=2)

#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='euclidean', knn=2)
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='manhattan', knn=2)
    A = affinity.get_affinity_matrix(proj, metric_method=distance.gaussian, knn=2)
#    A = affinity.get_affinity_matrix(proj, metric_method=distance.cosdist, knn=2)

    D = affinity.get_degree_matrix(A)
    
    print A
    print D
