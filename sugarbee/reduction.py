# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
import math
import numpy as np
from sklearn.decomposition import RandomizedPCA
import matplotlib

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

    sums = [0]*len(gmm.means_)
    for mi, _ in enumerate(gmm.means_) :
        m1 = gmm.means_[mi]
        c1 = gmm.covars_[mi]
        w1 = gmm.weights_[mi]
        ys = matplotlib.mlab.normpdf(value,m1,np.sqrt(c1))[0]*w1
        sums[mi] = ys[0]

#    score = max(np.median(sums), minval)
    score = np.median(sums)

    if score == 0:
        score = 1e-20
    score = -math.log(score)
    print score
    return score

def gmm_reduction(df, headers, gmms):
    proj = []
    gmm_normals = gmms[0]
    gmm_abnormals = gmms[1]

    for di, d in df.iterrows() :
#        print str(di) + "/" + str(len(df))
        normal_score = 0
        abnormal_score = 0

        normal_scores = []
        abnormal_scores = []

        protocol_type = d["protocol_type"]
        gmm_normals_prtcl = gmm_normals[protocol_type]
        gmm_abnormals_prtcl = gmm_abnormals[protocol_type]
        for hi, header in enumerate(headers) :
            if header in ["attack", "difficulty"] :
                continue
            val = d[header]
            gmm_normal_prtcl = gmm_normals_prtcl[hi]
            gmm_abnormal_prtcl = gmm_abnormals_prtcl[hi]
            if gmm_normal_prtcl == None :
                score = 0
            else :
                score = gmm_normal_prtcl.score([val]).tolist()[0]
#            score = get_score(gmm_normal_prtcl, val);
            normal_scores.append(score)
            if gmm_normal_prtcl == None :
                score = 0
            else :
                score = gmm_abnormal_prtcl.score([val]).tolist()[0]
#            score = get_score(gmm_abnormal_prtcl, val);
            abnormal_scores.append(score)

        normals = sum(normal_scores)
        abnormals = sum(abnormal_scores)

        scores = [max(normals,-20), max(abnormals,-20)]
        proj.append(scores)

    n_components = 2
    random_state = 0
    #rpca = RandomizedPCA(n_components=n_components, random_state=random_state)
    #proj = rpca.fit_transform(proj)
    return proj


def reduction(df,n_components=2, random_state=0):
    rpca = RandomizedPCA(n_components=n_components, random_state=random_state)
    proj = rpca.fit_transform(df)
    return proj

