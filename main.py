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
plot_lim_max = 30
plot_lim_min = -30

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
#    A = affinity.get_affinity_matrix(cproj, metric_method=distance.dist, metric_param='euclidean', knn=8)

    k = predict_k(A)
    logger.debug("supposed k : " + str(k))

    lim = 10; #int(len(df) * 0.01)
    if k == 1 :
        k = lim
    if k > lim :
        k = lim
    logger.debug("Total number of clusters : " + str(k))

    logger.debug(A)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    res = sc.labels_
#    logger.debug(res)

    clusters = [0] * k
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1

    workpath = os.path.dirname(os.path.abspath(__file__))

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
        df2 = df2 #[0:50]
        df = pd.concat([df1, df2])
        title = dataset_description + "_" + attack_type
        logger.debug("#################################################")
        logger.debug(title)
        logger.debug(str(len(df1)))
        logger.debug(str(len(df2)))
        test_clustering(df, gmms, title=title, save_to_file=True, highlight_point=attack_type_index)

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)
