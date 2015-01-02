# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import os
import numpy as np
import copy
import cPickle as pickle

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model
from nslkdd.get_kdd_dataframe import attack_types
from nslkdd.get_kdd_dataframe import df_by_attack_type
import sugarbee.reduction as reduction
import sugarbee.distance as distance
import sugarbee.affinity as affinity
import sugarbee.solver as solver

from autosp import predict_k
import util
import logger

today = util.make_today_folder('./results')
k = 10
num_clusters = k * 2

def print_confusion_matrix(true_values, clusters, res, highlight_point=None):
#""" Print confusion matrix in log file
#Param :
#true_values : true label per each dataset
#clusters : classified as normal or not
#res : result from kmeans
    #highlight_point
#"""
    # confusion matrix
    y_true = []
    y_pred = []

    for i in true_values :
        if i == model.attack_normal :
            y_true.append(0)
        else :
            y_true.append(1)

    for i in res :
        if clusters[i] >= 0 :
            y_pred.append(0)
        else :
            y_pred.append(1)

    m = confusion_matrix(list(y_true), list(y_pred))

    s1 = m[0][0] + m[0][1]
    s2 = m[1][1] + m[1][0]

    logger.debug(m)
    logger.debug("true_positive : " + str(m[0][0]) + " (" + str(m[0][0]*1.0 / s1) + ")")
    logger.debug("true_negative : " + str(m[1][1]) + " (" + str(m[1][1]*1.0 / s2) + ")")
    logger.debug("false_positive : " + str(m[1][0]) + " (" + str(m[1][0]*1.0 / s2) + ")")
    logger.debug("false_negative : " + str(m[0][1]) + " (" + str(m[0][1]*1.0 / s1) + ")")

    if highlight_point != None :
        # confusion matrix
        y_true = []
        y_pred = []
    
        for i, v in enumerate(true_values) :
            if v == highlight_point :
                y_true.append(1)
                if clusters[ res[i] ] >= 0 :
                    y_pred.append(0)
                else :
                    y_pred.append(1)
    
        logger.debug("")
        logger.debug("highlight")
        try :
            m = confusion_matrix(list(y_true), list(y_pred))
            s1 = m[0][0] + m[0][1]
            s2 = m[1][1] + m[1][0]
            logger.debug(m)
            logger.debug("true_positive : " + str(m[0][0]) + " (" + str(m[0][0]*1.0 / s1) + ")")
            logger.debug("true_negative : " + str(m[1][1]) + " (" + str(m[1][1]*1.0 / s2) + ")")
            logger.debug("false_positive : " + str(m[1][0]) + " (" + str(m[1][0]*1.0 / s2) + ")")
            logger.debug("false_negative : " + str(m[0][1]) + " (" + str(m[0][1]*1.0 / s1) + ")")

        except IndexError :
            logger.debug(y_true)
            logger.debug(y_pred)

def do_one_clustering(df, gmms):
    df_train = copy.deepcopy(df)
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    cproj = copy.deepcopy(proj)
    A = affinity.get_affinity_matrix(cproj, metric_method=distance.cosdist, knn=8)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    res = sc.labels_
    return res, cproj

def divide_df(df, res, cproj, true_attack_types):
    df_train = copy.deepcopy(df)
    res = res.tolist()

    clusters = [0] * k
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1

    normal = []
    abnormal = []
    t = dict(zip(set(res), map(lambda y: [i for i,z in enumerate(res) if z is y ], set(res))))
    for i in range(k):
        if clusters[i] > 0 :
            normal = normal + t[i]
        else :
            abnormal = abnormal + t[i]
    return normal, abnormal

def do_clustering(df, gmms, title="", save_to_file=False, highlight_point=None):
    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)
    true_attack_types = df["attack"].values.tolist()

    res1, cproj1 = do_one_clustering(df_train, gmms)
    res1 = res1.tolist()
    t1 = dict(zip(set(res1), map(lambda y: [i for i,z in enumerate(res1) if z is y ], set(res1))))
    df_train2 = df_train.iloc[t1[0],:]
    df_train3 = df_train.iloc[t1[1],:] # because iloc only get data with relative-index not static-index

    logger.debug("res1")
    logger.debug(res1)
    logger.debug("t1")
    logger.debug(t1)
    logger.debug("df_train1")
    logger.debug(df_train1.shape)
    logger.debug("df_train2")
    logger.debug(df_train2.shape)
    logger.debug("df_train3")
    logger.debug(df_train3.shape)

    try : 
        res2, cproj2 = do_one_clustering(df_train2, gmms)
    except : 
        res2 = [0] * df_train2.shape[0]
    try : 
        res3, cproj3 = do_one_clustering(df_train3, gmms)
    except : 
        res3 = [0] * df_train3.shape[0]
    res2 = res2.tolist()
    res3 = res3.tolist()

    logger.debug("res2")
    logger.debug(res2)
    logger.debug("res3")
    logger.debug(res3)

    res = []
    for i, j in df.iterrows() :
        try :
            df_train2.loc[i]
            print "##### pick #### " + str(i)
            pp = 0
            for q, _ in df_train2.iterrows() :
                if q == i :
                    res.append(res2[pp])
                    break
                pp = pp + 1
        except KeyError :
            pass
        try :
            df_train3.loc[i]
            print "///// pick //// " + str(i)
            pp = 0
            for q, _ in df_train3.iterrows() :
                if q == i :
                    res.append(res3[pp]+2)
                    break
                pp = pp + 1
        except KeyError :
            pass
    for i in res1 :
        res.append(res1[i]+4)
    logger.debug("res")
    logger.debug(res)

    logger.debug("res:")
    logger.debug(res)
    cproj = cproj1

    clusters = [0] * num_clusters
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1
    print_confusion_matrix(true_attack_types, clusters, res, highlight_point)

    print "save to file..." + title
    with open(today + "/" + title + '_cproj.pkl','wb') as output:
        pickle.dump(cproj, output, -1)
    with open(today + '/./' + title + '_res.pkl','wb') as output:
        pickle.dump(res, output, -1)
    with open(today + '/./' + title + '_df.pkl','wb') as output:
        pickle.dump(df, output, -1)
    with open(today + '/./' + title + '_highlight_point.pkl','wb') as output:
        pickle.dump(highlight_point, output, -1)


def test_clustering(df, gmms, title="", save_to_file=False, highlight_point=None):
    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    # from about 30 dimension to 2 dimension
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    cproj = copy.deepcopy(proj)

    # data_per_true_labels : try to make sort of dictionary per each label
    data_per_true_labels = []
    for i in range( len(attacks) ):
        data_per_true_labels.append([])

    true_attack_types = df["attack"].values.tolist()
    for i, d in enumerate(cproj):
        data_per_true_labels[true_attack_types[i]].append(d)

    A = affinity.get_affinity_matrix(cproj, metric_method=distance.cosdist, knn=8)

    k = predict_k(A)
    logger.debug("supposed k : " + str(k))

    lim = int(len(df) * 0.01)
    if lim < 3 or lim > 10 :
        lim = 10
    k = lim
    logger.debug("Total number of clusters : " + str(k))

    logger.debug(A)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    res = sc.labels_
    logger.debug(res)

    clusters = [0] * k
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1

    print_confusion_matrix(true_attack_types, clusters, res, highlight_point)

    logger.debug("Cluster count")
    counts = [0] * k
    for _, c in enumerate(res):
        counts[c] = counts[c] + 1
    logger.debug(str(counts))

    print "save to file..." + title
    with open(today + "/" + title + '_cproj.pkl','wb') as output:
        pickle.dump(cproj, output, -1)
    with open(today + '/./' + title + '_res.pkl','wb') as output:
        pickle.dump(res, output, -1)
    with open(today + '/./' + title + '_df.pkl','wb') as output:
        pickle.dump(df, output, -1)
    with open(today + '/./' + title + '_highlight_point.pkl','wb') as output:
        pickle.dump(highlight_point, output, -1)

if __name__ == '__main__':
    """ Anomaly detection with spectral clustering algorithm.
    First training set only, to see what would happen with only known classes
    Next with test set, to see what would happen with only unknown classes
    """
    import time
    start = time.time()

    headers, attacks = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    logger.set_file(today + "/log_main.txt")

    # with training-set
    gmms = gmms_20
    df1 = df_training_20[0:500]

    title = "training20_only"
    logger.debug("#################################################")
    logger.debug(title)
    test_clustering(df1, gmms, title=title, save_to_file=True)

    # with test-set
    dataset_description = "training20_test20"
    for attack_type_index, attack_type in enumerate(model.attack_types) :
        if attack_type_index == model.attack_normal : # why <= instead of ==
            continue
        df2 = df_by_attack_type(df_test_20, attack_type_index)
        df2 = df2[0:50]
        df = pd.concat([df1, df2])
        title = dataset_description + "_" + attack_type
        logger.debug("#################################################")
        logger.debug(title)
        logger.debug(str(len(df1)))
        logger.debug(str(len(df2)))
        test_clustering(df, gmms, title=title, save_to_file=True, highlight_point=attack_type_index)

    elapsed = (time.time() - start)
    logger.debug("done in %s seconds" % (elapsed))
