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

def show_one_row():
    df_train = copy.deepcopy(df)

    print len(gmms[0])
    print len(gmms[1])
    print len(gmms[0][0])

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

def show_classes():
    import os
    from nslkdd.data import model
    workpath = os.path.dirname(os.path.abspath(__file__))

    datafile_20 = workpath + '/nslkdd/data/KDDTrain+_20Percent.txt'
    datafile_full = workpath + '/nslkdd/data/KDDTrain+.txt'

    datafile_21 = workpath + '/nslkdd/data/KDDTest-21.txt'
    datafile_plus = workpath + '/nslkdd/data/KDDTest+.txt'

    headers, _ = preprocessing.get_header_data()

    dfs = []
    dfs.append(model.load_dataframe(datafile_20,headers))
    dfs.append(model.load_dataframe(datafile_full,headers))
    dfs.append(model.load_dataframe(datafile_21,headers))
    dfs.append(model.load_dataframe(datafile_plus,headers))


    df = dfs[0]
    df = df.iloc[[1,3],:]
    print df

    exit()


    # it shows every headers
#    for di, df in enumerate(dfs[0]) :
#        print df

    attacks = []
    for df in dfs :
        attacks.append( list(set(df['attack'])) )
    #    print attacks[-1]
    only_in_test_data = []
    for i in attacks[3] :
        if i in attacks[1] :
            pass
        else :
            only_in_test_data.append(i)
    total_test_set = attacks[1] + only_in_test_data
    print total_test_set

    # basic
    for di, df in enumerate(dfs) :
        print "=====" + str(di) + "======="
        s = 0 
        for i in total_test_set :
            det = len(df[df['attack']==i])
            s = s + det
            print i + " : " + str(len (df[df['attack']==i]))
        print "------------------"
        print "total : " + str(s)

    print "============================"
    # for tiddly
    df_names = ["Training_20", "Training_full", "Test_21", "Test_plus"]
    import copy
    for attack_type in total_test_set :
        for di, df_orig in enumerate(dfs) :
            df = copy.deepcopy(df_orig)
            df = df[df['attack'] == attack_type]
            category_name = str(list(set(df['protocol_type'])))
            df_name = df_names[di]
            perc = len(df) / (len(dfs[di])*1.0) * 100.0
            count = str(len(df)) + " / " + str(len(dfs[di])) + " (" + "{0:.3f}%".format(perc) + ")"
            bg = " "
            if perc == 0 :
                bg = "bgcolor(#cd5c5c): "
            print "| ! " + attack_type + " |" + bg + category_name + " |" + bg + df_name + " |" + bg + str(count) + " |" + bg + " |"

if __name__ == '__main__':
    show_classes()

