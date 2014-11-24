""" 
Data loader class for KDD99 dataset
"""

import time
import os
import math
import numpy as np
import pandas as pd

attack_types = ['guess_passwd', 'spy', 'ftp_write', 'nmap', 'back', 'multihop', 
    'rootkit', 'pod', 'portsweep', 'perl', 'ipsweep', 'teardrop', 'satan', 
    'loadmodule', 'buffer_overflow', 'normal', 'phf', 'warezmaster', 'imap', 
    'warezclient', 'land', 'neptune', 'smurf', 'processtable', 'named', 
    'udpstorm', 'snmpguess', 'sqlattack', 'ps', 'httptunnel', 'sendmail', 
    'snmpgetattack', 'apache2', 'saint', 'mailbomb', 'mscan', 'xterm', 'worm', 
    'xlock', 'xsnoop']

protocol_types = ["udp", "icmp", "tcp"]

attack_normal = 15

def load_headers(headerfile):
    """Headers, Attacks loader.
    
    It loads data from the file, but for attacks, hardcoded since many of those 
    attacks are not included in the header file.
    """

    assert(os.path.isfile(headerfile))
    with open(headerfile,'r') as f:
        _ = f.readline().replace('.\n','').split(',')
        headers = f.readlines()
    f.close()
    headers = [h.split(':')[0] for h in headers]
    headers.append('attack')
    headers.append('difficulty')
    return headers, attack_types

def load_dataframe(datafile,headers,datasize=None):
    assert(os.path.isfile(datafile))
    df = pd.read_csv(datafile, names=headers, nrows=datasize)
    df.iloc[0]
    return df

def discretize_attack_to_integer(df):
    for i in range(len(attack_types)):
        df['attack'][df['attack']==attack_types[i]]=i
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
    headers, attacks = load_headers(headerfile)
    df = load_dataframe(datafile,headers)
    elapsed = (time.time() - start)
    print "Total data : len=%d in %s seconds" % (len(df),elapsed)

    start = time.time()
    print "discretizing attack..."
    df = discretize_attack_to_integer(df)
    print "discretizing protocol type..."

    for i, r in df[10:20].iterrows():
        print "value " + str(r['protocol_type'])
    df = discretize_protocol_type_to_integer(df)
    for i, r in df[10:20].iterrows():
        print "value " + str(r['protocol_type'])

    print "discretizing service..."
    df = discretize_service_to_integer(df)
    print "discretizing flag..."
    df = discretize_flag_to_integer(df)
    scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised',
    'num_file_creations','count','srv_count','dst_host_count', 
    'dst_host_srv_count']
    print "scaling big numbers..."
    df = scale_bignums_to_log2(df,scaled)
    elapsed = (time.time() - start)
    print "pre-processing work done in %s seconds" % (elapsed)
    print df[:5]

    for i in range(len(headers)):
        print headers[i], max(df[headers[i]]), min(df[headers[i]])

