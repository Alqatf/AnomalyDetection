# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import cPickle as pickle
import numpy as np
import copy

import sugarbee.reduction as reduction

import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

if __name__ == '__main__':
    attack_names = ("back","buffer_overflow","ftp_write","guess_passwd","imap",
    "ipsweep","land","loadmodule","multihop","neptune",
    "nmap","normal","perl","phf","pod",
    "portsweep","rootkit","satan","smurf","spy",
    "teardrop","warezclient","warezmaster")
    colormaps = ["b","g","r","c","m","k","w","0.20","0.75","#eeefff",
    "#000fff","#235234","#345454","#5766723","#263543","#078787","#567576","#745655","#958673","#262434",
    "#dd2453","#eee253","#fff332"]
    df = None
    headers = None
    with open('df.pkl','rb') as input:
        df = pickle.load(input)
    with open('headers.pkl','rb') as input:
        headers = pickle.load(input)

    df_train = copy.deepcopy(df)
    df_train.drop('attack',1,inplace=True)
    df_train.drop('difficulty',1,inplace=True)

    proj = reduction.reduction(df_train, n_components=3)

    lists = []
    for i in range(22):
        lists.append([])

    attacks = df["attack"].values.tolist()

    for i, d in enumerate(proj):
        lists[attacks[i]].append(d)

    for i, proj in enumerate(lists) :
#        if len(proj) == 0:
#            continue
        x = [t[0] for t in proj]
        y = [t[1] for t in proj]
        x = np.array(x)
        y = np.array(y)
        colors = []
        for _ in range(len(x)):
            colors.append(colormaps[i])
        plt.scatter(x, y, c=colors)

#    plt.legend(attack_names, loc='best')

    plt.show()
