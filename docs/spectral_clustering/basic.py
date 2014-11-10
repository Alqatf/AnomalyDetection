import numpy as np

W = np.matrix([
[0,0.8,0.6,0,0.1,0],
[0.8,0,0.8,0,0,0],
[0.6,0.8,0,0.2,0,0],
[0,0,0.2,0,0.8,0.7],
[0.1,0,0,0.8,0,0.8],
[0,0,0,0.7,0.8,0],
])

D = np.zeros((6,6))
D[0,0] = 1.5
D[1,1] = 1.6
D[2,2] = 1.6
D[3,3] = 1.7
D[4,4] = 1.7
D[5,5] = 1.5

# Unnomalized
L = D - W
U, S, V = np.linalg.svd(L, full_matrices=True)
target = np.argmax(np.absolute(np.diff(S)))
print V.transpose()[:,target+1]

# Normalized
D[D == 0] = 1e-8 #don't laugh, there is a core package in R actually does this
D_hat = D**(-0.5)
L = D_hat * (D - W) * D_hat

U, S, V = np.linalg.svd(L, full_matrices=True)
target = np.argmax(np.absolute(np.diff(S)))
print V.transpose()[:,target+1]
