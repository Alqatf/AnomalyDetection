# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import copy

import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import sugarbee.reduction as reduction

if __name__ == '__main__':
    attack_names = ("back","buffer_overflow","ftp_write","guess_passwd","imap",
    "ipsweep","land","loadmodule","multihop","neptune",
    "nmap","normal","perl","phf","pod",
    "portsweep","rootkit","satan","smurf","spy",
    "teardrop","warezclient","warezmaster")

    colormaps = ["b","g","r","c","m","k","w","0.20","0.75","#eeefff",
    "#000fff","#235234","#345454","#5766723","#263543","#078787","#567576","#745655","#958673","#262434",
    "#dd2453","#eee253","#fff332"]

    import time
    start = time.time()

    df, headers, gmms = preprocessing.get_preprocessed_data()
    df = df[1000:2000]

    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')
#    headers = ['is_guest_login','wrong_fragment','num_compromised','dst_host_srv_count','logged_in','dst_bytes','dst_host_same_src_port_rate','srv_count','flag','protocol_type','src_bytes','count','service']
#    headers = ['dst_host_srv_count','dst_bytes','dst_host_same_src_port_rate','srv_count','src_bytes','count','service', 'dst_host_count']

    print "reductioning..."
    proj = reduction.gmm_reduction(df_train, headers, gmms)
    print "plotting..."
    lists = []
    for i in range(22):
        lists.append([])
    attacks = df["attack"].values.tolist()

    for i, d in enumerate(proj):
        lists[attacks[i]].append(d)

    plt.subplot(3, 1, 1)
    for i, proj in enumerate(lists) :
        if i == 11:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
            plt.plot(x, y, 'go')

    plt.subplot(3, 1, 2)
    for i, proj in enumerate(lists) :
        if i != 11:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
            plt.plot(x, y, 'ro')

    plt.subplot(3, 1, 3)
    for i, proj in enumerate(lists) :
        if i == 11:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
#            plt.plot(x, y, 'go')
            plt.scatter(x, y, c=colors)
        if i != 11:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
#            plt.plot(x, y, 'ro')
            plt.scatter(x, y, c=colors)


#    plt.legend(attack_names, loc='best')
    elapsed = (time.time() - start)

    print "done in %s seconds" % (elapsed)

    plt.show()
