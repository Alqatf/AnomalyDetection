"""

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics

"""

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


if __name__ == '__main__':
    a = np.matrix([1, 1, 1, 1, 1, 1, 1, 1, 1])
    b = np.matrix([1, 1, 1, 0, 1, 1, 1, 1, 0])
    print a
    print b
    print("cosine_similarity(a,b) %f" %( cosine_similarity(a,b)))

