# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.manifold import spectral_embedding
from sklearn.utils import check_random_state

import nslkdd.preprocessing as preprocessing
import sugarbee.reduction as reduction
import sugarbee.distance as distance
import sugarbee.affinity as affinity
import sugarbee.solver as solver

#def assign_undirected_weight(W, i, j, v):
#    W[i,j] = W[j,i] = v

if __name__ == '__main__':
    datasize = 30
    df, headers = preprocessing.get_preprocessed_data(datasize)
    proj = reduction.reduction(df, n_components=2)

    A = affinity.get_affinity_matrix(proj, metric_method=distance.dist, metric_param='euclidean', knn=3)
    D = affinity.get_degree_matrix(A)
    L = affinity.get_laplacian_matrix(A,D)

    X = solver.solve(L)

    est = KMeans(n_clusters=2)
    est.fit(X)
    print est.labels_

#    random_state = check_random_state(None)
#    maps = spectral_embedding(L, n_components = 3, 
#        eigen_solver = None, 
#        random_state = random_state,
#        eigen_tol = 0.0,
#        drop_first= False)
#    print maps
#    
#    _, labels, _ = k_means(maps,n_clusters=2,random_state=random_state,n_init=10)
#    print labels



