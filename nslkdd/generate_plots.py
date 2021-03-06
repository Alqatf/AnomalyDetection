"""
===========================================
Realisation plotting for data check
===========================================

In order to understand the data more, it will visualise 20% dataset for every 
properties.

It usually takes about 2026 seconds (35 minutes)
"""
print (__doc__)

import datetime
import math
import numpy as np
from sklearn import mixture
import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

import colorhex
from data import model
import preprocessing

def generate_realisation_plots(df, headers, path="", title="", gmms=None):
    min_sample = len(df)

    discretes = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attemped', 'num_root', 'num_file_creation', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

    for gpi, gmm_for_protocol_type in enumerate(gmms) :
        for hi, key in enumerate(headers):
            print "generating " + title + " figure for " + str(key) + " in protocol " + str(gpi) + " ..."
            fig, ax = plt.subplots()
            ax.grid(True)
            plt.title(key + " " + str(len(df)))
            plt.ylabel(key + " (logscale)")

            df_for_protocl = df[df['protocol_type']==gpi]
            df_for_protocl = df_for_protocl[key]
    
            if len(df_for_protocl) == 0 :
                continue

            minval = min(df_for_protocl)
            maxval = max(df_for_protocl)
            margin = 0 #(maxval-minval) / 20.0
    
            # gmm fitting
            clf = gmms[gpi][hi]
            if None == clf :
                continue
    
            # show realisation
            plt.subplot(3, 1, 1)
            plt.xlabel(key)
            plt.ylabel("count")
    
            xs = range(len(df_for_protocl))
            plt.scatter(xs, df_for_protocl)
    
            # show hist
            plt.subplot(3, 1, 2)
            plt.xlabel(key)
            plt.ylabel("count")
    
            binwidth = -1
            if key in discretes :
                binwidth = int(abs(maxval - minval + 1)) * 2
            else :
                binwidth = 10 * 2
            histdist = df_for_protocl.hist(bins=30)
    #        histdist = plt.hist(df_for_protocl, binwidth, normed=1, facecolor='b')
    
            # show GMM
            plt.subplot(3, 1, 3)
            plt.xlabel(key)
            plt.ylabel('prob')
    
            gmm_sample = min_sample
            x = histdist
            xs = np.linspace(minval-margin,maxval+margin,gmm_sample)
    
            # gmm draw
            yss = [0]*len(xs)
            colors = colorhex.codes
            for mi, _ in enumerate(clf.means_) :
                m1 = clf.means_[mi]
                c1 = clf.covars_[mi]
                w1 = clf.weights_[mi]
    
                ys = [matplotlib.mlab.normpdf(x,m1,np.sqrt(c1))[0]*w1 for x in xs]
                for yi,y in enumerate(ys) :
                    yss[yi] = yss[yi] + y
            plt.plot(xs,yss,color='y', lw=5)
    
            for mi, _ in enumerate(clf.means_) :
                m1 = clf.means_[mi]
                c1 = clf.covars_[mi]
                w1 = clf.weights_[mi]
    
                ys = [matplotlib.mlab.normpdf(x,m1,np.sqrt(c1))[0]*w1 for x in xs]
                plt.plot(xs,ys,color=colors[mi], lw=1)
    
            # save and close
            fig.savefig("./plots/" + path + "/" + key+ "_prtcl_" + str(gpi) + "_" + title + "_" + path + ".png")
            plt.cla() # http://stackoverflow.com/questions/8213522/matplotlib-clearing-a-plot-when-to-use-cla-clf-or-close/8228808#8228808
            plt.clf()
            plt.close('all')

def draw_gmm(df, gmms, headers, path) :
    normal_gmm = gmms[0]
    abnormal_gmm = gmms[1]

    # plot for abnormal
    df_train = copy.deepcopy(df)
    df_train = df_train[(df_train["attack"] != model.attack_normal)] # only select for normal data
    df_train.drop('attack',1,inplace=True) # remove useless 
    df_train.drop('difficulty',1,inplace=True) # remove useless 
    df_train.reset_index(drop=True)
    generate_realisation_plots(df_train, headers, path=path, title="abnormal", gmms=abnormal_gmm)

    # plot for normal
    df_train = copy.deepcopy(df)
    df_train = df_train[(df_train["attack"] == model.attack_normal)] # only select for normal data
    df_train.drop('attack',1,inplace=True) # remove useless 
    df_train.drop('difficulty',1,inplace=True) # remove useless 
    df_train.reset_index(drop=True)
    df_train.reset_index(drop=True)
    generate_realisation_plots(df_train, headers, path=path, title="normal", gmms=normal_gmm)

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

    headers, _ = preprocessing.get_header_data()
    headers.remove('protocol_type')
    headers.remove('attack')
    headers.remove('difficulty')

    # plot preparation
    df_training_20, df_training_full, gmms_20, gmms_full = preprocessing.get_preprocessed_training_data(datasize)
    draw_gmm(df_training_20, gmms_20, headers, "training20")
#    draw_gmm(df_training_full, gmms_full, headers, "trainingfull")
#
#    # plot preparation
#    df_test_plus, df_test_21, gmms_test_plus, gmms_test_21 = preprocessing.get_preprocessed_test_data(datasize)
#    draw_gmm(df_test_plus, gmms_test_plus, headers, "testplus")
#    draw_gmm(df_test_21, gmms_test_21, headers, "test21")

    elapsed = (time.time() - start)
    print "Plotting done (%s seconds)" % (elapsed)

