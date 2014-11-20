# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
import numpy as np
from sklearn.decomposition import RandomizedPCA

def get_score(gmm, value):
    minval = 1e+20
    minidx = -1

    # one of distribution
    # or density of distributions
    for mi, _ in enumerate(gmm.means_):
        det = abs(mi - value)
        m1 = gmm.means_[mi]
        c1 = gmm.covars_[mi]
        if minval > det :
            minval = matplotlib.mlab.normpdf(value,m1,np.sqrt(c1))[0][0]
    minval = minval*len(gmm.means_) # make it dense

    sums = 0
    for mi, _ in enumerate(gmm.means_) :
        m1 = gmm.means_[mi]
        c1 = gmm.covars_[mi]
        w1 = gmm.weights_[mi]
        ys = matplotlib.mlab.normpdf(value,m1,np.sqrt(c1))[0]*w1
        sums = sums + ys[0]

    score = max(sums, minval)

    if score == 0:
        score = 1e-20
    score = math.log(score)

    return score

def gmm_reduction(df, headers, gmms):
    proj = []
    gmm_normals = gmms[0]
    gmm_abnormals = gmms[1]

    for di, d in df.iterrows() :
        print str(di) + "/" + str(len(df))
        normal_score = 0
        abnormal_score = 0

        normal_scores = []
        abnormal_scores = []
        for hi, header in enumerate(headers) :
            if header in ["attack", "difficulty"] :
                continue
            val = d[header]
            gmm_normal = gmm_normals[hi]
            gmm_abnormal = gmm_abnormals[hi]
            score = gmm_normal.score([val]).tolist()[0]
            normal_scores.append(score)
            score = gmm_abnormal.score([val]).tolist()[0]
            abnormal_scores.append(score)

#        print str(sum(normal_scores)) + " vs " + str(max(normal_scores))
#        print str(sum(abnormal_scores)) + " vs " + str(max(abnormal_scores))
#        normal_score = sum(normal_scores)
#        abnormal_score = sum(abnormal_scores)

        normals = sum(normal_scores)
        abnormals = sum(abnormal_scores)

        scores = [max(normals,-50), max(abnormals,-50)]
        proj.append(scores)

    n_components = 2
    random_state = 0
    rpca = RandomizedPCA(n_components=n_components, random_state=random_state)
    #proj = rpca.fit_transform(proj)
    return proj


def reduction(df,n_components=2, random_state=0):
    rpca = RandomizedPCA(n_components=n_components, random_state=random_state)
    proj = rpca.fit_transform(df)
    return proj

