import numpy as np

def get_ith_eigenvector(V,i):
    n = len(V)
    ei = n-i-1
    return V.transpose()[:,ei]

def solve(L, k=2):
    U, S, V = np.linalg.svd(L, full_matrices=True)
    target = np.argmax(np.absolute(np.diff(S)))
#    print V.transpose()[:,target+1]

    n = len(L)
    eigenspace = np.zeros((n, k));

    for i in range(k):
        eigenspace[:,i] = get_ith_eigenvector(V,i)

    X = np.array(eigenspace)
    return X

