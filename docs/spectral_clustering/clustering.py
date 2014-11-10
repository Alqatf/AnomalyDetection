# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import k_means

def assign_undirected_weight(W, i, j, v):
    W[i,j] = W[j,i] = v

def unnormalized_L(D,W):
    L = D-W
    return L

def normalized_L(D,W):
    D[D == 0] = 1e-8 #don't laugh, there is a core package in R actually does this
    D_hat = D**(-0.5)
    L = D_hat * (D - W) * D_hat
#    D_hat = D**(-1)
#    L = D_hat * (D - W)
    return L

def get_second_eigenvector(V):
    return get_ith_eigenvector(V,1) # zero-based

# zero-based
def get_ith_eigenvector(V,i):
    ei = n-i-1
    return V.transpose()[:,ei]

def get_ideal_k(S):
    target = np.argmax(np.absolute(np.diff(S)))
    return target + 2

def get_svd(L):
    U, S, V = np.linalg.svd(L, full_matrices=True)
    U[np.absolute(U) < 1e-8] = 0
    S[np.absolute(S) < 1e-8] = 0
    V[np.absolute(V) < 1e-8] = 0
    return U, S, V

if __name__ == '__main__':
    n = 5;
    V = range(n)
    
    W = np.zeros((n,n))
    assign_undirected_weight(W,0,1,0.08)
    assign_undirected_weight(W,0,2,0.09)
    assign_undirected_weight(W,1,2,0.45)
    assign_undirected_weight(W,1,3,0.22)
    assign_undirected_weight(W,1,4,0.24)
    assign_undirected_weight(W,2,3,0.2)
    assign_undirected_weight(W,2,4,0.19)
    assign_undirected_weight(W,3,4,1)
    
    D = np.zeros((n,n))
    for i in V:
        D[i,i] = np.sum(W[i,:])
    
    U,S,V = get_svd(unnormalized_L(D,W))
    print get_second_eigenvector(V)
    U,S,V = get_svd(normalized_L(D,W))
    print get_second_eigenvector(V)
    
    
    k = get_ideal_k(S)
    eigenspace = np.zeros((n, k));
    for i in range(k):
        eigenspace[:,i] = get_ith_eigenvector(V,i)
    
    X = np.array(eigenspace)
    print V
    print X
    

    est = KMeans(n_clusters=2)
    plt.cla()
    est.fit(X)
    print est.labels_

    _, labels, _ = k_means(X, n_clusters=2)
    print labels
    exit()
    fignum = 1
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    cluster_centers = est.cluster_centers_
    ax.scatter(X[:, 1], X[:, 0], X[:, 2])
    cluster_center = cluster_centers[0]
    ax.plot(cluster_centers[:,1], cluster_centers[:,0], cluster_centers[:,2], 'x', markerfacecolor='#4EACC5',markeredgecolor='k', markersize=6)
    
    plt.show()
