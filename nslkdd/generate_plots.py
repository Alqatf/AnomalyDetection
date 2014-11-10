"""
===========================================
Realisation plotting for data check
===========================================

In order to understand the data more, it will visualise 20% dataset for every properties.

It usually takes 
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

from data import model
import preprocessing

def generate_realisation_plots(df, headers):
    for key in headers:
        fig, ax = plt.subplots()
        ax.autoscale_view()
        ax.grid(True)
        plt.title(key + " " + str(len(df)))
        plt.ylabel(key + " (logscale)")

        # gmm fitting
        clf = mixture.GMM(n_components=2, covariance_type='full', min_covar=0.00001)
        clf.fit(df[key])
        m1, m2 = clf.means_
        w1, w2 = clf.weights_
        c1, c2 = clf.covars_

        # show hist
        histdist = plt.hist(df[key], 5, normed=True, alpha=0.2)

        # show gausses
        plotgauss1 = lambda x: plt.plot(x,w1*matplotlib.mlab.normpdf(x,m1,np.sqrt(c1))[0], linewidth=2, color='k')
        plotgauss2 = lambda x: plt.plot(x,w2*matplotlib.mlab.normpdf(x,m2,np.sqrt(c2))[0], linewidth=2, color='r')
        plotgauss1(histdist[1])
        plotgauss2(histdist[1])

        # save and close
        fig.savefig("./plots/realisation_"+key+"_log.png")
        plt.close()

if __name__ == '__main__':
    import sys
    import time

    if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
        print ("Requires Python 2.7.x")
        exit()
    del sys

    datasize = None
    start = time.time()
    df, headers = preprocessing.get_preprocessed_data(datasize)
    generate_realisation_plots(df, headers)
    elapsed = (time.time() - start)
    print "[histogram] done in %s seconds" % (elapsed)

