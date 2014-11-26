"""
It usually takes about 8296 seconds (2 hours 20 minutes).
"""

import os
import math
import copy
import cPickle as pickle
import numpy as np
from data import model
from sklearn import mixture

def discretize_elems_in_list(df, l, key):
    """
    It converts categorical values to integer values, so you need to care 
    when you use this function.
    """

    for i, _ in enumerate(l):
        df[key][df[key]==l[i]]=i
    return df

def scale_bignums_to_log2(df,targets):
    for i in range(len(targets)):
        for j in range(df.shape[0]):
                df[targets[i]].iloc[j]=int(math.log(1+df[targets[i]].iloc[j],2))
    df.iloc[0]
    return df

def discretize_elems(df, attacks):
    protocols = list(set(df['protocol_type']))
    services = list(set(df['service']))
    flags = list(set(df['flag']))

    df = discretize_elems_in_list(df,attacks,'attack')
    df = discretize_elems_in_list(df,protocols,'protocol_type')
    df = discretize_elems_in_list(df,services,'service')
    df = discretize_elems_in_list(df,flags,'flag')

    scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised',
    'num_file_creations','count','srv_count','dst_host_count', 
    'dst_host_srv_count']
    df = scale_bignums_to_log2(df,scaled)
    return df

def generate_gmms(df, headers, n_initialization=10):
    """
    Using BIC, AIC values may be required for future work
    """
    gmms = [] # it is for normal data or abnormal data.

    """
    Three elements for each protocol_type, and in each element, there is a 
    container for each headers.
    """
    for protocol_type in range(3): #0:udp, 1:icmp, 2:tcp
        df_for_protocol = df[ df['protocol_type']==protocol_type ]
        gmms_for_protocol = []

        for header_type in headers:
            if header_type  in ['protocol_type', 'attack', 'difficulty']:
                continue

            # pick the best clf which produces minimum bic among those four types
            data_to_fit = df_for_protocol[header_type]
            cov_types = ['spherical', 'tied', 'diag', 'full']
            best_clf = None
            lowest_bic = np.infty
            if len(data_to_fit) != 0:
                # If there is no data, it become None type.
                print header_type
                for cov_type in cov_types:
                    try :
                        # gmm fitting
                        clf = mixture.GMM(n_components=5,
                            covariance_type=cov_type,
                            n_init=n_initialization)
    
                        clf.fit(data_to_fit)
                        bic = clf.bic(data_to_fit)
                        if bic < lowest_bic:
                            best_clf = clf
                            lowest_bic = bic
                    except :
                        print "     Warning! " + header_type + " w/" + cov_type + " has an error."
                        pass
                print lowest_bic
            gmms_for_protocol.append(best_clf)
        gmms.append(gmms_for_protocol)
    return gmms

def construct_gmms(df, headers):
    # only select for normal data
    df_train = copy.deepcopy(df)
    df_train = df_train[(df_train["attack"] == model.attack_normal)] 
    gmms_normal = generate_gmms(df_train, headers)

    # only select for abnormal data
    df_train = copy.deepcopy(df)
    df_train = df_train[(df_train["attack"] != model.attack_normal)] 
    gmms_abnormal = generate_gmms(df_train, headers)

    gmms = [gmms_normal, gmms_abnormal]
    return gmms

def get_header_data():
    workpath = os.path.dirname(os.path.abspath(__file__))
    headerfile = workpath + '/data/kddcup.names'
    headers, attacks = model.load_headers(headerfile)
    return headers, attacks

def get_preprocessed_test_data(datasize=None, regenerate=False):
    df = None
    workpath = os.path.dirname(os.path.abspath(__file__))

    if regenerate == False:
        with open(workpath+'/./df_test_plus.pkl','rb') as input:
            df_test_plus = pickle.load(input)
        with open(workpath+'/./df_test_21.pkl','rb') as input:
            df_test_21 = pickle.load(input)
        with open(workpath + '/./gmms_test_plus.pkl','rb') as input: 
            gmm_test_plus = pickle.load(input)
        with open(workpath + '/./gmms_test_21.pkl','rb') as input: 
            gmm_test_21 = pickle.load(input)
        return df_test_plus, df_test_21, gmm_test_plus, gmm_test_21
    else : 
        workpath = os.path.dirname(os.path.abspath(__file__))
        datafile_plus = workpath + '/data/KDDTest+.txt'
        datafile_21 = workpath + '/data/KDDTest-21.txt'

        headers, attacks = get_header_data()

        print "preprocessing testing data plus..."
        df = model.load_dataframe(datafile_plus,headers,datasize=datasize)
        df_test_plus = discretize_elems(df, attacks)
        gmms_test_plus = construct_gmms(df_test_plus, headers)

        print "preprocessing testing data 21..."
        df = model.load_dataframe(datafile_21,headers,datasize=datasize)
        df_test_21 = discretize_elems(df, attacks)
        gmms_test_21 = construct_gmms(df_test_21, headers)

        print "save to file..."
        with open(workpath + '/./df_test_plus.pkl','wb') as output:
            pickle.dump(df_test_plus, output, -1)
        with open(workpath + '/./df_test_21.pkl','wb') as output:
            pickle.dump(df_test_21, output, -1)
        with open(workpath + '/./gmms_test_plus.pkl','wb') as output:
            pickle.dump(gmms_test_plus, output,-1)
        with open(workpath + '/./gmms_test_21.pkl','wb') as output:
            pickle.dump(gmms_test_21, output,-1)

        return df_test_plus, df_test_21, gmms_test_plus, gmms_test_21

def get_preprocessed_training_data(datasize=None, regenerate=False, withfull=False):
    df = None
    workpath = os.path.dirname(os.path.abspath(__file__))

    if regenerate == False:
        with open(workpath+'/./df_training_20.pkl','rb') as input:
            df_training_20 = pickle.load(input)
        with open(workpath+'/./df_training_full.pkl','rb') as input:
            df_training_full = pickle.load(input)
        with open(workpath+'/./gmms_training_20.pkl','rb') as input:
            gmms_20 = pickle.load(input)
        with open(workpath+'/./gmms_training_full.pkl','rb') as input:
            gmms_full = pickle.load(input)
        return df_training_20, df_training_full, gmms_20, gmms_full

    else : 
        workpath = os.path.dirname(os.path.abspath(__file__))
        datafile_20 = workpath + '/data/KDDTrain+_20Percent.txt'
        datafile_full = workpath + '/data/KDDTrain+.txt'

        headers, attacks = get_header_data()

        df_training_full = None
        gmms_training_full = None

        print "preprocessing training data for 20 percent..."
        df = model.load_dataframe(datafile_20,headers,datasize=datasize)
        print "descretization..."
        df_training_20 = discretize_elems(df, attacks)
        print "gmm fitting..."
        gmms_training_20 = construct_gmms(df_training_20, headers)

        if withfull == True : 
            print "preprocessing training data total..."
            df = model.load_dataframe(datafile_full,headers,datasize=datasize)
            print "descretization..."
            df_training_full = discretize_elems(df, attacks)
            print "gmm fitting..."
            gmms_training_full = construct_gmms(df_training_full, headers)
        else :
            print "without full data"

        print "save to file..."
        with open(workpath + '/./df_training_20.pkl','wb') as output:
            pickle.dump(df_training_20, output,-1)
        with open(workpath + '/./gmms_training_20.pkl','wb') as output:
            pickle.dump(gmms_training_20, output,-1)
        if withfull == True :
            with open(workpath + '/./df_training_full.pkl','wb') as output:
                pickle.dump(df_training_full, output,-1)
            with open(workpath + '/./gmms_training_full.pkl','wb') as output:
                pickle.dump(gmms_training_full, output,-1)

        return df_training_20, df_training_full, gmms_training_20, gmms_training_full

if __name__ == '__main__':
    import sys
    import time

    if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
        print ("Requires Python 2.7.x")
        exit()
    del sys

    print (__doc__)

    #############################################################################
    # Generate pkl files 
    datasize = None
    start = time.time()

    #############################################################################
    print "preprocessing training data..."
    get_preprocessed_training_data(datasize,regenerate=True, withfull=False)

    #############################################################################
    print "preprocessing test data..."
    get_preprocessed_test_data(datasize,regenerate=True)

    #############################################################################
    elapsed = (time.time() - start)
    print "Preprocessing done (%s seconds)" % (elapsed)

