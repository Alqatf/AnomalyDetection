import math
from data import model
import numpy as np

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

def get_preprocessed_data(datasize):
    headerfile = './data/kddcup.names'
    datafile = './data/KDDTrain+_20Percent.txt'

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

    return df, headers

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.pyplot
    import matplotlib.mlab

    df = get_preprocessed_data()
    headerfile = './data/kddcup.names'

    headers, _ = model.load_headers(headerfile)
    key = 'duration'
    histdist = plt.hist(df[key], 30, normed=True, alpha=0.2)

    plt.show()

