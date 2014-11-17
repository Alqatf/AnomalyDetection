import distance
import numpy as np
import copy
import sugarbee.distance as distance

def assign_undirected_weight(M, i, j, v):
    M[i,j] = M[j,i] = v

def get_degree_matrix(A):
    n = len(A)
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(A[i,:])
    return D

def get_affinity_matrix(proj, metric_method=distance.gaussian, metric_param=None, knn=8):
    n = len(proj)
    S = np.zeros((n,n))
    A = np.zeros((n,n))

    # similarity matrix
    for i in range(n):
        for j in range(i+1,n):
            dist = metric_method(proj[i], proj[j], metric_param)
            assign_undirected_weight(S,i,j,dist)

    # affinity matrix
    if knn < n:
        for i in range(n):
            row = copy.deepcopy(S[i][:])
            row.sort()
            row = row[::-1] # reserve sort
            row = row[:knn]
            for j in range(n):
                if S[i,j] in row:
                    A[i,j] = S[i,j]
    else :
        A = S

    return A

def get_laplacian_matrix(A,D):
    print A
    print D
    D[D == 0] = 1e-8 #don't laugh, there is a core package in R actually does this
    D_hat = D**(-0.5)
    L = D_hat * (D-A) * D_hat
    return D-A

