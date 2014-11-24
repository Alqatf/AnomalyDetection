# -*- coding: utf-8 -*-
"""
It generates plots that shows similarity for anomalies in each dataset.
"""

import copy
import math
import numpy as np
import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model

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
    minval = minval*len(gmm.means_)

    sums = 0 
    for mi, _ in enumerate(gmm.means_) :
        m1 = gmm.means_[mi]
        c1 = gmm.covars_[mi]
        w1 = gmm.weights_[mi]
        ys = matplotlib.mlab.normpdf(value,m1,np.sqrt(c1))[0]*w1
        sums = sums + ys[0]

#    if sums > minval :
#        print "=== sums ==="
#    else :
#        print "=== minval ==="
#    print minval
#    print sums

    score = max(sums, minval) 

    if score == 0:
        score = 1e-20
#    print "score : " + str(score)
    score = math.log(score)
    return score

def generate_plots(df_abnormal, df_normal, headers, gmms, title, path="", protcls_name=""):
    proj = []

    gmm_normals = gmms[0]
    gmm_abnormals = gmms[1]

    fig, ax = plt.subplots()
    plt.subplot(2, 1, 1)
    plt.title("normal scores")
    plt.subplot(2, 1, 2)
    plt.title("abnormal scores")

    for di, d in df_normal.iterrows() :
#        print str(di) + "/" + str(len(df_normal))
        normal_score = 0
        abnormal_score = 0
        normal_scores = []
        abnormal_scores = []
        for hi, header in enumerate(headers) :
            if header in ["protocol_type", "attack", "difficulty"] :
                continue
            val = d[header]
            gmm_normal = gmm_normals[hi]
            gmm_abnormal = gmm_abnormals[hi]
            score = get_score(gmm_normal,val)
            normal_scores.append(score)
            score = get_score(gmm_abnormal,val)
            abnormal_scores.append(score)
        xs = range(len(headers))
        plt.subplot(2, 1, 1)
        plt.plot(xs,normal_scores,color='y', lw=3)
        plt.subplot(2, 1, 2)
        plt.plot(xs,abnormal_scores,color='y', lw=3)

    for di, d in df_abnormal.iterrows() :
        print str(di) + "/" + str(len(df_abnormal))
        normal_score = 0
        abnormal_score = 0
        normal_scores = []
        abnormal_scores = []
        for hi, header in enumerate(headers) :
            if header in ["protocol_type", "attack", "difficulty"] :
                continue
            val = d[header]
            gmm_normal = gmm_normals[hi]
            gmm_abnormal = gmm_abnormals[hi]
            score = get_score(gmm_normal,val)
            normal_scores.append(score)
            score = get_score(gmm_abnormal,val)
            abnormal_scores.append(score)

        xs = range(len(headers))
        plt.subplot(2, 1, 1)
        plt.plot(xs,normal_scores,color='b', lw=1)
        plt.subplot(2, 1, 2)
        plt.plot(xs,abnormal_scores,color='b', lw=1)

    # save and close
    filename = "./plots/" + path + "/" + title + "_" + protcls_name + "_" + path + ".png"
    print filename 
    fig.savefig(filename)
    plt.close()

def generate_plots_for_df(df, gmms, path="") :
    headers, _ = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    # plot for classes
    protocol_types = model.protocol_types #["udp","tcp","icmp"]

    for protocol_index, protocol_type in enumerate(protocol_types):
        gmm_normals = gmms[0][protocol_index]
        gmm_abnormals = gmms[1][protocol_index]

        # normal data
        df_normal = copy.deepcopy(df)
        df_normal = df_normal[(df_normal["attack"] == 11)] # only select for 1 class 
        df_normal = df_normal[(df_normal["protocol_type"] == protocol_index)]
        df_normal.drop('attack',1,inplace=True) # remove useless 
        df_normal.drop('difficulty',1,inplace=True) # remove useless 
        df_normal.drop('protocol_type',1,inplace=True)
        df_normal.reset_index(drop=True)
        df_normal = df_normal[0:10]

        # abnormal data
        for i, attack_type in enumerate(model.attack_types) :
            if i == 11 :
                continue
            df_abnormal = copy.deepcopy(df)
            df_abnormal = df_abnormal[(df_abnormal["attack"] == i)] # only select for 1 class 
            df_abnormal = df_abnormal[(df_abnormal["protocol_type"] == protocol_index)]

            if 1 >  len(df_abnormal) :
                continue

            df_abnormal.drop('attack',1,inplace=True) # remove useless 
            df_abnormal.drop('difficulty',1,inplace=True) # remove useless 
            df_abnormal.drop('protocol_type',1,inplace=True)
            df_abnormal.reset_index(drop=True)
            df_abnormal = df_abnormal[0:10]

            gmm_normals_protcl = gmms[0][protocol_index]
            gmm_abnormals_protcl = gmms[1][protocol_index]
            gmms_protcl = [gmm_normals_protcl, gmm_abnormals_protcl]

            generate_plots(df_abnormal, df_normal, headers, gmms_protcl, attack_type, path=path, protcls_name = protocol_type)

if __name__ == '__main__':
    import time
    start = time.time()

    df_training_20, df_training_full, gmms_training_20, gmms_training_full = preprocessing.get_preprocessed_training_data()
    df_test_plus, df_test_21, gmms_test_plus, gmms_test_21 = preprocessing.get_preprocessed_test_data()

    generate_plots_for_df(df_training_20, gmms_training_20, "training20")
    generate_plots_for_df(df_training_full, gmms_training_full, "trainingfull")
    generate_plots_for_df(df_test_plus, gmms_test_plus, "testplus")
    generate_plots_for_df(df_test_21, gmms_test_21, "test21")

