# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html

Also this document helps me much.
https://www.kaggle.com/c/titanic-gettingStarted/details/getting-started-with-python-ii
"""
print (__doc__)

import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.manifold import spectral_embedding
from sklearn.utils import check_random_state

import nslkdd.preprocessing as preprocessing

if __name__ == '__main__':
    datasize = 10
    df, headers, _ = preprocessing.get_preprocessed_data(datasize)
    df_train = copy.deepcopy(df)
    
    # select only normal data
    df_train = df_train[(df_train["attack"] == 11)]

    # remove every attack and difficulty
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    # select one row
    print df_train[0:1]
    
    for i, r in df_train.iterrows() :
        # show one value
        print "value " + str(r['duration'])
