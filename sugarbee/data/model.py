""" Test code for existing dataset
Author : Kim Seonghyun <shyeon.kim@scipi.net>
"""

import time
import os
import math
import numpy as np
import pandas as pd

def load_headers(headerfile):
    assert(os.path.isfile(headerfile))
    with open(headerfile,'r') as f:
        attacks=f.readline().replace('.\n','').split(',')
        headers=f.readlines()
    f.close()
    headers = [h.split(':')[0] for h in headers]
    headers.append('attack')
    headers.append('attack_idx')
    return headers, attacks

def load_dataframe(datafile,headers,datasize=None):
    assert(os.path.isfile(datafile))
    df = pd.read_csv(datafile, names=headers, nrows=datasize)
    df.iloc[0]
    return df

def discretize_attack_to_integer(df):
    for i in range(len(attacks)):
        df['attack'][df['attack']==attacks[i]]=i
    return df

def discretize_protocol_type_to_integer(df):
    pt=list(set(df['protocol_type']))
    for i in range(len(pt)):
        df['protocol_type'][df['protocol_type']==pt[i]]=i
    return df

def discretize_service_to_integer(df):
    pt=list(set(df['service']))
    for i in range(len(pt)):
        df['service'][df['service']==pt[i]]=i
    return df

def discretize_flag_to_integer(df):
    pt=list(set(df['flag']))
    for i in range(len(pt)):
        df['flag'][df['flag']==pt[i]]=i
    return df

def scale_bignums_to_log2(df,targets):
    for i in range(len(targets)):
        for j in range(df.shape[0]):
                df[targets[i]].iloc[j]=int(math.log(1+df[targets[i]].iloc[j],2))
    df.iloc[0]
    return df

if __name__ == '__main__':
    import sys
    if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
        print ("Requires Python 2.7.x")
        exit()
    del sys

    datafile = './KDDTrain+_20Percent.txt'

    start = time.time()
    headerfile = './kddcup.names'
    headers,attacks = load_headers()
    df = load_dataframe(datafile,headers)
    elapsed = (time.time() - start)
    print "data len=%d in %s seconds" % (len(df),elapsed)

    start = time.time()
    df = discretize_attack_to_integer(df)
    df = discretize_protocol_type_to_integer(df)
    df = discretize_service_to_integer(df)
    df = discretize_flag_to_integer(df)
    scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised','num_file_creations','count','srv_count','dst_host_count', 'dst_host_srv_count']
    df = scale_bignums_to_log2(df,scaled)
    elapsed = (time.time() - start)
    print "pre-processing work done in %s seconds" % (elapsed)
    print df[:5]

    for i in range(len(headers)):
        print headers[i], max(df[headers[i]])
