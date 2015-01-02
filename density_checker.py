# -*- coding: utf-8 -*-

import numpy as np
import copy

import pandas as pd
import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
import colorhex
import util
import logger


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
#    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    logger.set_file(today + "/log_rep.txt")
    # with training-set
    df1 = df_training_20
    gmms = gmms_20
    title = "training20_only"
    logger.debug("#################################################")
    logger.debug(title)
    test_clustering(df1, gmms, title=title, save_to_file=True)
