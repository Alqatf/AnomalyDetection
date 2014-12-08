import numpy as np

n = 6;
V = range(n)

W = np.matrix([
[0,0.8,0.6,0,0.1,0],
[0.8,0,0.8,0,0,0],
[0.6,0.8,0,0.2,0,0],
[0,0,0.2,0,0.8,0.7],
[0.1,0,0,0.8,0,0.8],
[0,0,0,0.7,0.8,0],
])

D = np.zeros((n,n))
for i in V:
    D[i,i] = np.sum(W[i,:])

# Unnomalized
L = D - W

# Normalized
D[D == 0] = 1e-8 #don't laugh, there is a core package in R actually does this
D_hat = D**(-0.5)
L = D_hat * (D - W) * D_hat

# Biclustering
U, S, V = np.linalg.svd(L, full_matrices=True)
target = np.argmax(np.absolute(np.diff(S)))
print target
print V
print V.transpose()[:,target+1]
