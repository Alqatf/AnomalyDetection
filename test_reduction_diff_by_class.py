# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing

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

def generate_plots(df_abnormal, df_normal, headers, gmms, title):
    proj = []
    gmm_normals = gmms[0]
    gmm_abnormals = gmms[1]

    fig, ax = plt.subplots()
    for di, d in df_normal.iterrows() :
        print str(di) + "/" + str(len(df_normal))
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
#            score = gmm_normal.score([val]).tolist()[0]
            score = get_score(gmm_normal,val)
#            print "normal : " + str(score)
            normal_scores.append(score)
#            score = gmm_abnormal.score([val]).tolist()[0]
            score = get_score(gmm_abnormal,val)
#            print "abnormal : " + str(score)
            abnormal_scores.append(score)
#            print "======================"
        xs = range(len(headers))
        plt.subplot(2, 1, 1)
        plt.plot(xs,normal_scores,color='y', lw=3)
        plt.subplot(2, 1, 2)
        plt.plot(xs,abnormal_scores,color='y', lw=3)

#    raw_input()
#    print "##############################"

    for di, d in df_abnormal.iterrows() :
        print str(di) + "/" + str(len(df_abnormal))
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
#            score = gmm_normal.score([val]).tolist()[0]
            score = get_score(gmm_normal,val)
#            print "normal : " + str(score)
            normal_scores.append(score)
#            score = gmm_abnormal.score([val]).tolist()[0]
            score = get_score(gmm_abnormal,val)
            abnormal_scores.append(score)
#            print "abnormal : " + str(score)
#            print "======================"

        xs = range(len(headers))
        plt.subplot(2, 1, 1)
        plt.plot(xs,normal_scores,color='b', lw=1)
        plt.subplot(2, 1, 2)
        plt.plot(xs,abnormal_scores,color='b', lw=1)

    # save and close
    fig.savefig("./plots/" + title + ".png")
    plt.close()

if __name__ == '__main__':
    import time
    start = time.time()

    df, headers, gmms = preprocessing.get_preprocessed_data()
    headers.remove('attack')
    headers.remove('difficulty')

    # plot for classes
    attack_names = ["back","buffer_overflow","ftp_write","guess_passwd","imap",
    "ipsweep","land","loadmodule","multihop","neptune",
    "nmap","normal","perl","phf","pod",
    "portsweep","rootkit","satan","smurf","spy",
    "teardrop","warezclient","warezmaster"]

    # normal data
    df_normal = copy.deepcopy(df)
    df_normal = df_normal[(df_normal["attack"] == 11)] # only select for 1 class 
    df_normal.drop('attack',1,inplace=True) # remove useless 
    df_normal.drop('difficulty',1,inplace=True) # remove useless 
    df_normal.reset_index(drop=True)
    df_normal = df_normal[0:10]

    # abnormal data
    for i, attack in enumerate(attack_names) :
        if i == 11 :
            continue
        print "================"
        print attack
        df_abnormal = copy.deepcopy(df)
        df_abnormal = df_abnormal[(df_abnormal["attack"] == i)] # only select for 1 class 
        df_abnormal.drop('attack',1,inplace=True) # remove useless 
        df_abnormal.drop('difficulty',1,inplace=True) # remove useless 
        df_abnormal.reset_index(drop=True)
        df_abnormal = df_abnormal[0:10]
        generate_plots(df_abnormal, df_normal, headers, gmms, attack)
