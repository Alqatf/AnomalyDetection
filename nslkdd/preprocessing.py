"""
It usually takes about 400 seconds (7 minutes)
"""
print (__doc__)

import os
import math
import copy
import cPickle as pickle
import numpy as np
from data import model
from sklearn import mixture

def discretize_elems_in_list(df, l, key):
    for i, _ in enumerate(l):
        df[key][df[key]==l[i]]=i
    return df

def scale_bignums_to_log2(df,targets):
    for i in range(len(targets)):
        for j in range(df.shape[0]):
                df[targets[i]].iloc[j]=int(math.log(1+df[targets[i]].iloc[j],2))
    df.iloc[0]
    return df

def get_preprocessed_data(datasize=None, headerfile = './nslkdd/data/kddcup.names', datafile = './nslkdd/data/KDDTrain+_20Percent.txt', regenerate=False):
    df = None
    headers = None
    workpath = os.path.dirname(os.path.abspath(__file__))
    print workpath
    if os.path.isfile(workpath+'/./df.pkl') and os.path.isfile(workpath+'/./headers.pkl') and os.path.isfile(workpath+'/./gmms.pkl') and regenerate == False:
        with open(workpath+'/./df.pkl','rb') as input:
            df = pickle.load(input)
        with open(workpath+'/./headers.pkl','rb') as input:
            headers = pickle.load(input)
        with open(workpath+'/./gmms.pkl','rb') as input:
            gmms = pickle.load(input)
        return df, headers, gmms

    else : 
        headers, attacks = model.load_headers(headerfile)
        df = model.load_dataframe(datafile,headers,datasize=datasize)
    
        protocols = list(set(df['protocol_type']))
        services = list(set(df['service']))
        flags = list(set(df['flag']))
    
        df = discretize_elems_in_list(df,attacks,'attack')
        df = discretize_elems_in_list(df,protocols,'protocol_type')
        df = discretize_elems_in_list(df,services,'service')
        df = discretize_elems_in_list(df,flags,'flag')
    
        scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised','num_file_creations','count','srv_count','dst_host_count', 'dst_host_srv_count']
        df = scale_bignums_to_log2(df,scaled)

        # only select for normal data
        df_train = copy.deepcopy(df)
        df_train = df_train[(df_train["attack"] == 11)] 
        gmms_normal = generate_gmms(df_train, headers)

        # only select for abnormal data
        df_train = copy.deepcopy(df)
        df_train = df_train[(df_train["attack"] != 11)] 
        gmms_abnormal = generate_gmms(df_train, headers)

        gmms = [gmms_normal, gmms_abnormal]

        with open(workpath + '/./df.pkl','wb') as output:
            pickle.dump(df, output,-1)
        with open(workpath + '/./headers.pkl','wb') as output:
            pickle.dump(headers, output,-1)
        with open(workpath + '/./gmms.pkl','wb') as output:
            pickle.dump(gmms, output,-1)

        return df, headers, gmms

def generate_gmms(df, headers, min_covariance=0.001, n_initialization=10):
    gmms = []
    for protocol_type in range(3): #udp, tcp, icmp
        df_for_protocol = df[df['protocol_type']==protocol_type]
        gmms_for_protocol = []
        for key in headers:
            if key  in ['protocol_type', 'attack', 'difficulty'] :
                continue
            # gmm fitting
            clf = mixture.GMM(n_components=5, covariance_type='full', min_covar=min_covariance, n_init=n_initialization)
            clf.fit(df_for_protocol[key])
            gmms_for_protocol.append(clf)
        gmms.append(gmms_for_protocol)
    return gmms

if __name__ == '__main__':
    import sys
    import time

    if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
        print ("Requires Python 2.7.x")
        exit()
    del sys

    headerfile = './data/kddcup.names'
    datafile = './data/KDDTrain+_20Percent.txt'

    datasize = None
    start = time.time()
    print "preprocessing data..."
    df, headers, _ = get_preprocessed_data(datasize,headerfile,datafile,regenerate=True)
    elapsed = (time.time() - start)
    print "Preprocessing done (%s seconds)" % (elapsed)

