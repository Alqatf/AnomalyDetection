# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html

Several papers suggest to use only several headers not the others. I did not count on it as it is not my interest.

headers = ['is_guest_login','wrong_fragment','num_compromised','dst_host_srv_count','logged_in','dst_bytes','dst_host_same_src_port_rate','srv_count','flag','protocol_type','src_bytes','count','service']
headers = ['dst_host_srv_count','dst_bytes','dst_host_same_src_port_rate','srv_count','src_bytes','count','service', 'dst_host_count']
"""
print (__doc__)

import numpy as np
import copy

import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model

import sugarbee.reduction as reduction
import colorhex

if __name__ == '__main__':
    import time
    start = time.time()

    colormaps = colorhex.codes

    headers, attacks = preprocessing.get_header_data()
    df_training_20, df_training_full, gmms_training_20, gmms_training_full = preprocessing.get_preprocessed_training_data()
    df_test_20, df_test_full, gmms_test_20, gmms_test_full = preprocessing.get_preprocessed_test_data()

    df = df_training_20
    gmms = gmms_training_20
    df = df[0:1000]

    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    print "reductioning..."
    proj = reduction.gmm_reduction(df_train, headers, gmms)

    print "plotting..."
    true_labels = []
    for i in range( len(attacks) ):
        true_labels.append([])

    attacks = df["attack"].values.tolist()

    for i, d in enumerate(proj):
        true_labels[attacks[i]].append(d)

    # title for the plots
    titles = ['Normal data',
              'Abnormal data',
              'Data']

    plt.subplot(3, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(-50, 100)
    plt.ylim(-50, 100)
    plt.title(titles[0])
#    plt.xticks(())
#    plt.yticks(())

    for i, proj in enumerate(true_labels) :
        if i == model.attack_normal:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
            plt.plot(x, y, 'go')

    plt.subplot(3, 1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(-50, 100)
    plt.ylim(-50, 100)
    plt.title(titles[1])
#    plt.xticks(())
#    plt.yticks(())

    for i, proj in enumerate(true_labels) :
        if i != model.attack_normal:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
            plt.plot(x, y, 'ro')

    plt.subplot(3, 1, 3)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(-50, 100)
    plt.ylim(-50, 100)
    plt.title(titles[2])

#    plt.xticks(())
#    plt.yticks(())

    for i, proj in enumerate(true_labels) :
        if i == model.attack_normal:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
#            plt.plot(x, y, 'go')
            plt.scatter(x, y, c=colors)
        if i != model.attack_normal:
            x = [t[0] for t in proj]
            y = [t[1] for t in proj]
            x = np.array(x)
            y = np.array(y)
            colors = []
            for _ in range(len(x)):
                colors.append(colormaps[i])
#            plt.plot(x, y, 'ro')
            plt.scatter(x, y, c=colors)

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)

    plt.show()
