# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import copy
import cPickle as pickle

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

import nslkdd.preprocessing as preprocessing
import nslkdd.data.model as model
from nslkdd.get_kdd_dataframe import attack_types
from nslkdd.get_kdd_dataframe import df_by_attack_type

import colorhex
import util

import logger

today = util.make_today_folder('./results')
#today = "./results/2015-01-08"

plot_lim_max = 21
plot_lim_min = -21

def plot_true_labels(ax, data_per_true_labels, title="", highlight_point = None):
    ax.set_title("True labels")
    for i, p in enumerate(data_per_true_labels) :
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        if i == model.attack_normal:
            colors = ['g'] * len(x)
            ax.scatter(x, y, c=colors)
        elif i != model.attack_normal and i != highlight_point:
            colors = ['r'] * len(x)
            ax.scatter(x, y, c=colors)

    if highlight_point != None :
        p = data_per_true_labels[highlight_point]
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        colors = ['y'] * len(x)
        ax.scatter(x, y, c=colors)

def plot_normal_label(ax, data_per_true_labels, title=""):
    ax.set_title(title)
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        if i == model.attack_normal:
            ax.scatter(x, y, c='g')
            logger.debug("* mean/std of normal")
            logger.debug(len(x))
            logger.debug(np.mean(x))
            logger.debug(np.mean(y))
            logger.debug(np.std(x))
            logger.debug(np.std(y))

def plot_abnormal_label(ax, data_per_true_labels, title=""):
    ax.set_title(title)
    for i, p in enumerate(data_per_true_labels) :
        x = [t[0] for t in p]
        y = [t[1] for t in p]
        x = np.array(x)
        y = np.array(y)
        if i != model.attack_normal:
            ax.scatter(x, y, c='r')

def get_data(title):
    with open(today+'/'+title+'_cproj.pkl','rb') as input:
        cproj = pickle.load(input)
    with open(today+'/'+title+'_res.pkl','rb') as input:
        res = pickle.load(input)
    with open(today+'/'+title+'_df.pkl','rb') as input:
        df = pickle.load(input)
    with open(today+'/'+title+'_highlight_point.pkl','rb') as input:
        highlight_point = pickle.load(input)
    return cproj, res, df, highlight_point

def gen_plot(cproj, res, df, highlight_point, title):
    _, attacks = preprocessing.get_header_data()

    # figure setting
    fig, axarr = plt.subplots(4, 4, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(plot_lim_min, plot_lim_max)
    plt.ylim(plot_lim_min, plot_lim_max)
#    plt.xlabel('interval')
#    plt.ylabel('log(probability) + k')
#    plt.title('Convergence plot')
#    plt.grid(True)

    data_per_true_labels = []
    for i in range( len(attacks) ):
        data_per_true_labels.append([])

    true_attack_types = df["attack"].values.tolist()
    for i, d in enumerate(cproj):
        data_per_true_labels[true_attack_types[i]].append(d)

    k = int( len(cproj) * 12/500.0)
    clusters = [0] * k
    cluster_xs = []
    cluster_ys = []
    for i in range(k):
        cluster_xs.append([])
        cluster_ys.append([])
    cluster_xmeans = [0] * k
    cluster_ymeans = [0] * k
    cluster_xstds = [0] * k
    cluster_ystds = [0] * k

    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1
        cluster_xs[ res[i] ].append(p[0])
        cluster_ys[ res[i] ].append(p[1])

    logger.debug("* mean/std of cluster")
    for i, cluster in enumerate(clusters) :
        cluster_xmeans[i] = np.mean(cluster_xs[i])
        cluster_ymeans[i] = np.mean(cluster_ys[i])
        cluster_xstds[i] = np.std(cluster_xs[i])
        cluster_ystds[i] = np.std(cluster_ys[i])
        logger.debug("cluster : " + str(i))
        logger.debug("- size [" + str(len(cluster_xs[i])) + "]")
        logger.debug("- xmin [" + str(cluster_xmeans[i]) + "]")
        logger.debug("- ymin [" + str(cluster_ymeans[i]) + "]")
        logger.debug("- xstd [" + str(cluster_xstds[i]) + "]")
        logger.debug("- ystd [" + str(cluster_ystds[i]) + "]")

    ax1 = axarr[0, 0]
    ax2 = axarr[0, 1]
    ax3 = axarr[0, 2]
    ax4 = axarr[0, 3]
    ax5 = axarr[1, 0]
    ax6 = axarr[1, 1]
    ax7 = axarr[1, 2]
    ax8 = axarr[1, 3]
    ax9 = axarr[2, 0]
    ax10 = axarr[2, 1]
    ax11 = axarr[2, 2]
    ax12 = axarr[2, 3]
    ax13 = axarr[3, 0]
    ax14 = axarr[3, 1]
    ax15 = axarr[3, 2]
    ax16 = axarr[3, 3]

    plot_true_labels(ax1, data_per_true_labels, "True labels", highlight_point)
    plot_normal_label(ax2, data_per_true_labels, "True normals")
    plot_abnormal_label(ax3, data_per_true_labels, "True abnormal")

    ax4.set_title("k-means")
    for i, p in enumerate(cproj):
        ax4.scatter(p[0], p[1], c=colorhex.codes[ res[i] ])
    ##############################################################
    ax5.set_title("Normal res")
    for i, p in enumerate(cproj):
        if clusters[ res[i] ] >= 0 :
            ax5.scatter(p[0], p[1], c='g')
    ##############################################################
    ax6.set_title("Abnormal res")
    for i, p in enumerate(cproj):
        if clusters[ res[i] ] < 0 :
            ax6.scatter(p[0], p[1], c='r')
    ##############################################################
    ax7.set_title("Cluster 1")
    for i, p in enumerate(cproj):
        if res[i] == 0 :
            ax7.scatter(p[0], p[1], c='g')
    ##############################################################
    ax8.set_title("Cluster 2")
    for i, p in enumerate(cproj):
        if res[i] == 1 :
            ax8.scatter(p[0], p[1], c='g')
    ##############################################################
#    ax9.set_title("kmeans")
#    kmean_plot(title, ax9)
    ##############################################################
    ax9.set_title("Cluster 3")
    for i, p in enumerate(cproj):
        if res[i] == 2 :
            ax9.scatter(p[0], p[1], c='g')
    ##############################################################
    ax10.set_title("Cluster 4")
    for i, p in enumerate(cproj):
        if res[i] == 3 :
            ax10.scatter(p[0], p[1], c='g')
    ##############################################################
    ax11.set_title("Cluster 5")
    for i, p in enumerate(cproj):
        if res[i] == 4 :
            ax11.scatter(p[0], p[1], c='g')
    ##############################################################
    ax12.set_title("Cluster 6")
    for i, p in enumerate(cproj):
        if res[i] == 5 :
            ax12.scatter(p[0], p[1], c='g')
    ##############################################################
    ax13.set_title("Cluster 7")
    for i, p in enumerate(cproj):
        if res[i] == 6 :
            ax13.scatter(p[0], p[1], c='g')
    ##############################################################
    ax14.set_title("Cluster 8")
    for i, p in enumerate(cproj):
        if res[i] == 7 :
            ax14.scatter(p[0], p[1], c='g')
    ##############################################################
    ax15.set_title("Cluster 9")
    for i, p in enumerate(cproj):
        if res[i] == 8 :
            ax15.scatter(p[0], p[1], c='g')
    ##############################################################
    ax16.set_title("Cluster 10")
    for i, p in enumerate(cproj):
        if res[i] == 9 :
            ax16.scatter(p[0], p[1], c='g')
    ##############################################################

    print title + " has been saved"
    fig.savefig(today + "/" + title + ".png")
    plt.close()

    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(plot_lim_min, plot_lim_max)
    plt.ylim(plot_lim_min, plot_lim_max)
    for i, p in enumerate(data_per_true_labels) :
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        if i == model.attack_normal:
            colors = ['g'] * len(x)
            ax.scatter(x, y, c=colors)
        elif i != model.attack_normal and i != highlight_point:
            colors = ['r'] * len(x)
            ax.scatter(x, y, c=colors)
    if highlight_point != None :
        p = data_per_true_labels[highlight_point]
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        colors = ['y'] * len(x)
        ax.scatter(x, y, c=colors)
    plt.xlabel('Similarity score to normal')
    plt.ylabel('Similarity score to abnormal')
    plt.title('True labels')
    plt.grid(True)
    fig.savefig(today + "/" + title + "_true_.png")
    plt.close()

    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(plot_lim_min, plot_lim_max)
    plt.ylim(plot_lim_min, plot_lim_max)
    for i, p in enumerate(cproj):
        if clusters[ res[i] ] >= 0 :
            ax.scatter(p[0], p[1], c='g')
        else :
            ax.scatter(p[0], p[1], c='r')
    plt.xlabel('Similarity score to normal')
    plt.ylabel('Similarity score to abnormal')
    plt.title('Prediected labels')
    plt.grid(True)
    fig.savefig(today + "/" + title + "_prediction_.png")
    plt.close()


def gen_plots():
    dataset_description = "training20_only"
    title = dataset_description
    cproj, res, df, highlight_point = get_data(title)
    gen_plot(cproj, res, df, highlight_point, title)

    dataset_description = "training20_test20"
    for attack_type_index, attack_type in enumerate(model.attack_types) :
        if attack_type_index == model.attack_normal : # why <= instead of !=
            continue
        title = dataset_description + "_" + attack_type
        cproj, res, df, highlight_point = get_data(title)
        gen_plot(cproj, res, df, highlight_point, title)

def gen_one_plot():
    dataset_description = "training20_test20_guess_passwd"
    title = dataset_description
    cproj, res, df, highlight_point = get_data(title)
    gen_plot(cproj, res, df, highlight_point, title)

def kmean_plot(title, ax):
    _, attacks = preprocessing.get_header_data()
    cproj, res, df, highlight_point = get_data(title)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
#    plt.xlim(plot_lim_min, plot_lim_max)
#    plt.ylim(plot_lim_min, plot_lim_max)
#    ax = axarr
#    ax.set_title("plot")

    data_per_true_labels = []
    for i in range( len(attacks) ):
        data_per_true_labels.append([])
    true_attack_types = df["attack"].values.tolist()
    for i, d in enumerate(cproj):
        data_per_true_labels[true_attack_types[i]].append(d)

    k = 10
    clusters = [0] * k
    for i, p in enumerate(cproj):
        true_label = true_attack_types[i]
        if true_label == model.attack_normal :
            clusters[ res[i] ] = clusters[ res[i] ] + 1
        else :
            clusters[ res[i] ] = clusters[ res[i] ] - 1

    x = []
    y = []
    p = []
    for ii, pp in enumerate(cproj):
        if clusters[ res[ii] ] > 0 :
            x.append(pp[0])
            y.append(pp[1])
            p.append(pp)

    from sklearn.cluster import KMeans
    data = p
    h = .02
    estimator = KMeans(init='k-means++', n_clusters=3)
    estimator.fit(data)
    centroids = estimator.cluster_centers_

    x_min, x_max = min(x) + 1, max(x) - 1
    y_min, y_max = min(y) + 1, max(y) - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect='auto', origin='lower')
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    colors = ['g'] * len(x)
    ax.scatter(x, y, c=colors)
    ax.scatter(np.mean(x), np.mean(y), c='r')
    ax.scatter(np.median(x), np.median(y), c='b')
    delta = 0.025
    X = np.arange(plot_lim_min, plot_lim_max, delta)
    Y = np.arange(plot_lim_min, plot_lim_max, delta)
    X,Y = np.meshgrid(X,Y)
    Z = mlab.bivariate_normal(X, Y, np.std(x), np.std(y), np.mean(x), np.mean(y))
    plt.contour(X,Y,Z)

def test():
    _, attacks = preprocessing.get_header_data()
    dataset_description = "training20_only"
    title = dataset_description
    cproj, res, df, highlight_point = get_data(title)

    fig, axarr = plt.subplots(1, 1, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(plot_lim_min, plot_lim_max)
    plt.ylim(plot_lim_min, plot_lim_max)
    ax = axarr
    ax.set_title("plot")

    data_per_true_labels = []
    for i in range( len(attacks) ):
        data_per_true_labels.append([])
    true_attack_types = df["attack"].values.tolist()
    for i, d in enumerate(cproj):
        data_per_true_labels[true_attack_types[i]].append(d)

    for i, p in enumerate(data_per_true_labels) :
        x = np.array([t[0] for t in p])
        y = np.array([t[1] for t in p])
        if i == model.attack_normal:
            from sklearn.cluster import KMeans
            data = p
            h = .02
            estimator = KMeans(init='k-means++', n_clusters=3)
            estimator.fit(data)
            centroids = estimator.cluster_centers_

            x_min, x_max = min(x) + 1, max(x) - 1
            y_min, y_max = min(y) + 1, max(y) - 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
#            plt.figure(1)
#            plt.clf()

            plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')
            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Paired,
                       aspect='auto', origin='lower')
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='x', s=169, linewidths=3,
                        color='w', zorder=10)

            colors = ['g'] * len(x)
            ax.scatter(x, y, c=colors)
            ax.scatter(np.mean(x), np.mean(y), c='r')
            ax.scatter(np.median(x), np.median(y), c='b')
            delta = 0.025
            X = np.arange(plot_lim_min, plot_lim_max, delta)
            Y = np.arange(plot_lim_min, plot_lim_max, delta)
            X,Y = np.meshgrid(X,Y)
            Z = mlab.bivariate_normal(X, Y, np.std(x), np.std(y), np.mean(x), np.mean(y))
            plt.contour(X,Y,Z)

#    for i, r in df.iterrows() :
#        if r['attack']
#    for i, p in enumerate(cproj):
#        if res[i] == 8 :
#            ax1.scatter(p[0], p[1], c='g')

#    plt.xticks(())
#    plt.yticks(())

    plt.show()
    plt.close()

if __name__ == '__main__':
    """ Anomaly detection with spectral clustering algorithm.
    First training set only, to see what would happen with only known classes
    Next with test set, to see what would happen with only unknown classes
    """
    import time
    start = time.time()

    logger.set_file(today + "/log_plots.txt")
    gen_plots()
    # gen_one_plot()
    # test()

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)
