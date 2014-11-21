# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import *
from scipy import *
from autosp import predict_k


def consistent_labels(labels):
    """Achieve "some" consistency of color between true labels and pred labels.


    Parameters
    ----------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    Returns
    ----------
    color_map : dict object {integer: integer}
        The map of labels.
    """

    color_map = {}

    i = 0
    v = 0
    while v != max(labels) + 1:
        if labels[i] in color_map:
            pass
        else:
            color_map[labels[i]] = v
            v += 1
        i += 1

    return color_map

if __name__ == "__main__":
    # Generate artificial datasets.
    number_of_blobs = 6  # You can change this!!

    datax = [0.3, 0.4, 0.6, 0.2, 0.5, 0.4]
    datay = [0.2, 0.5, 0.4, 0.6, 0.2, 0.6]

    # Calculate affinity_matrix.
    affinity_matrix = csr_matrix( (6,6), dtype=float )
    affinity_matrix[0,1] = 0.8
    affinity_matrix[1,0] = 0.8
    affinity_matrix[0,2] = 0.6
    affinity_matrix[2,0] = 0.6
    affinity_matrix[0,4] = 0.1
    affinity_matrix[4,0] = 0.1
    affinity_matrix[1,2] = 0.8
    affinity_matrix[2,1] = 0.8
    affinity_matrix[2,3] = 0.2
    affinity_matrix[3,2] = 0.2
    affinity_matrix[3,4] = 0.8
    affinity_matrix[4,3] = 0.8
    affinity_matrix[3,5] = 0.7
    affinity_matrix[5,3] = 0.7
    affinity_matrix[4,5] = 0.8
    affinity_matrix[5,4] = 0.8

    print affinity_matrix.todense()

    # auto_spectral_clustering
    k = predict_k(affinity_matrix)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(affinity_matrix)

    labels_pred = sc.labels_

    print("%d blobs(artificial datasets)." % number_of_blobs)
    print("%d clusters(predicted)." % k)

    # Plot.
    from pylab import *
    labels_true = [0,0,0,1,1,1]
    t_map = consistent_labels(labels_true)
    t = [t_map[v] for v in labels_true]

    p_map = consistent_labels(labels_pred)
    p = [p_map[v] for v in labels_pred]

    print labels_pred
    print p_map

    print t
    print p

    subplot(211)
    title("%d blobs(artificial datasets)." % number_of_blobs)
    scatter(datax, datay, s=150, c=t)

    subplot(212)
    title("%d clusters(predicted)." % k)
    scatter(datax, datay, s=150, c=p)

    show()
