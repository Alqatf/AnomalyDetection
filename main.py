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
import matplotlib.mlab as mlab

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
k = 12
plot_lim_max = 21
plot_lim_min = -21

def check_abnormal_with_density(meanx, meany, stdx, stdy, target_sz):
    normal_sz = 264
    target_sz = target_sz
    X = np.arange(plot_lim_min, plot_lim_max, 0.1)
    Y = np.arange(plot_lim_min, plot_lim_max, 0.1)

    mX, mY = np.meshgrid(X, Y)
    n1 = mlab.bivariate_normal(mX, mY, 2.15, 0.89, 15.31, -6.5)
    n2 = mlab.bivariate_normal(mX, mY, 3.16, 3.21, 18, -17.5)
    n3 = mlab.bivariate_normal(mX, mY, 1.79, 1, 17.5, -11.3)
    n4 = mlab.bivariate_normal(mX, mY, 3.6, 2.5, 16, -11.4)
    n5 = mlab.bivariate_normal(mX, mY, 3.4, 2.6, 14, -18)
    n6 = mlab.bivariate_normal(mX, mY, 3.65, 4.85, 11, 4.7)
    n7 = mlab.bivariate_normal(mX, mY, 1.85, 1.45, 15.8, -13)
    normal_dist = n1 + n2 + n3 + n4 + n5 + n6 + n7
    target_dist = mlab.bivariate_normal(mX, mY, stdx, stdy, meanx, meany)

    normal_dist = normal_dist*(normal_sz/float(normal_sz+target_sz))
    target_dist = target_dist*(target_sz/float(target_sz+target_sz))

    s = 0
    for x in range(len(X)) :
        for y in range(len(Y)):
            det = target_dist[x,y] - normal_dist[x,y]
            if det > 0:
                s = s + det
    return s

def print_confusion_matrix(true_values, clusters, res, highlight_point,
        clusters_xmean, clusters_ymean, clusters_xstd, clusters_ystd) :
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
    # preprocessing
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

#    lim = int(len(df) * 0.01)
#    lim = 12
#    if lim < 3 or lim > 10 :
#        lim = 10
    lim = int( len(proj) * 12/500.0  )
    k = lim
    logger.debug("Total number of clusters : " + str(k))

    logger.debug(A)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(A)
    res = sc.labels_
    logger.debug(res)

    # cluster data set
    clusters = [0] * k
    clusters_data = []
    clusters_xmean = [-1] * k
    clusters_ymean = [-1] * k
    clusters_xstd = [-1] * k
    clusters_ystd = [-1] * k
    for i in range(k) :
        clusters_data.append([])
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1
        clusters_data[ res[i] ].append(p)

    # cluster recheck with density
    for i, cluster in enumerate(clusters) :
        p = clusters_data[i]
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        clusters_xmean[i] = np.mean(x)
        clusters_ymean[i] = np.mean(y)
        clusters_xstd[i] = np.std(x)
        clusters_ystd[i] = np.std(y)

    ds = []
    for i, cluster in enumerate(clusters) :
        if cluster > 0 :
            d = check_abnormal_with_density(clusters_xmean[i],
                clusters_ymean[i],
                clusters_xstd[i],
                clusters_ystd[i],
                len(clusters_data[i]))
            ds.append(d)
            if 0 > d:
                clusters[i] = -99999
        else :
            ds.append(None)
    logger.debug("ds")
    logger.debug(ds)

    # report
    print_confusion_matrix(true_attack_types, clusters, res, highlight_point,
        clusters_xmean, clusters_ymean, clusters_xstd, clusters_ystd)

    logger.debug("Clusters")
    logger.debug(clusters)
    counts = [0] * k
    for _, c in enumerate(res):
        counts[c] = counts[c] + 1
    logger.debug("Cluster datacount")
    logger.debug(str(counts))

    # save to file
    print "save to file..." + title
    with open(today + "/" + title + '_cproj.pkl','wb') as output:
        pickle.dump(cproj, output, -1)
    with open(today + '/./' + title + '_res.pkl','wb') as output:
        pickle.dump(res, output, -1)
    with open(today + '/./' + title + '_df.pkl','wb') as output:
        pickle.dump(df, output, -1)
    with open(today + "/" + title + '_clusters_xmean.pkl','wb') as output:
        pickle.dump(clusters_xmean, output, -1)
    with open(today + "/" + title + '_clusters_ymean.pkl','wb') as output:
        pickle.dump(clusters_ymean, output, -1)
    with open(today + "/" + title + '_clusters_xstd.pkl','wb') as output:
        pickle.dump(clusters_xstd, output, -1)
    with open(today + "/" + title + '_clusters_ystd.pkl','wb') as output:
        pickle.dump(clusters_ystd, output, -1)
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
    df1 = df_training_20[0:2000]

    title = "training20_only"
    logger.debug("#################################################")
    logger.debug(title)
    test_clustering(df1, gmms, title=title, save_to_file=True, highlight_point=None)

    # with test-set
    dataset_description = "training20_test20"
    for attack_type_index, attack_type in enumerate(model.attack_types) :
        if attack_type_index == model.attack_normal : # why <= instead of ==
            continue
        df2 = df_by_attack_type(df_test_20, attack_type_index)
        df2 = df2[0:100]
        df = pd.concat([df1, df2])
        title = dataset_description + "_" + attack_type
        logger.debug("#################################################")
        logger.debug(title)
        logger.debug(str(len(df1)))
        logger.debug(str(len(df2)))
        test_clustering(df, gmms, title=title, save_to_file=True, highlight_point=attack_type_index)

    elapsed = (time.time() - start)
    logger.debug("done in %s seconds" % (elapsed))
