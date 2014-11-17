# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.graph import graph_laplacian

def assign_undirected_weight(W, i, j, v):
    W[i,j] = W[j,i] = v

n = 5
W = np.zeros((n,n))
assign_undirected_weight(W,0,1,0.08)
assign_undirected_weight(W,0,2,0.09)
assign_undirected_weight(W,1,2,0.45)
assign_undirected_weight(W,1,3,0.22)
assign_undirected_weight(W,1,4,0.24)
assign_undirected_weight(W,2,3,0.2)
assign_undirected_weight(W,2,4,0.19)
assign_undirected_weight(W,3,4,1)

adjacency = W;
print W
laplacian, dd = graph_laplacian(adjacency, normed=True, return_diag=True)

print laplacian
print dd
