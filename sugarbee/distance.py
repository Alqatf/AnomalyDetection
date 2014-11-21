import math
import numpy as np
from sklearn.neighbors import DistanceMetric
from scipy.spatial.distance import mahalanobis;
from sklearn.metrics.pairwise import cosine_similarity

# https://spectrallyclustered.wordpress.com/2010/06/05/sprint-1-k-means-spectral-clustering/
def gaussian(x,y,param=None):
    diffs = []
    for i in range(len(x)):
        diff = (x[i] - y[i]) ** 2
        diffs.append(diff)
    total_diff = -1 * sum(diffs)
    val = (((np.median(diffs) * 1.5) ** 2) * 2)
    c=2*((1.5 * np.median(diffs))**2)

    val = total_diff / c
    return math.exp(val)

def dist(X_1, X_2, param='euclidean'):
    dist = DistanceMetric.get_metric(param)
    X = [X_1,X_2]
    return dist.pairwise(X)[0,1]

#def madist(x, y, invcovar_):
#    return np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i]))
#
#def invcovar(l):
#    covariance_xy = np.cov(x,y, rowvar=0)
#    inv_covariance_xy = np.linalg.inv(covariance_xy)
#    xy_mean = np.mean(x),np.mean(y)
#    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
#    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
#    diff_xy = np.transpose([x_diff, y_diff])
#    return inv_covariance_xy, diff_xy
#
def cosdist(x, y, param=None):
    return cosine_similarity(x,y)
#
#X_1 = [1,2,2,3,4,5]
#X_2 = [4,5,5,5,3,2]
#X_3 = [4,5,5,5,3,2]
#X_4 = [2,4,5,2,3,2]
#X_5 = [2,5,5,5,3,2]
#X_6 = [1,5,5,5,3,2]
#
#print dist(X_1,X_2,'euclidean')
#print dist(X_1,X_2,'manhattan')
##print madist(X_1,X_2,invcovar([X_1,X_2,X_3,X_4,X_5,X_6]))
#print cosdist(X_1,X_2)
