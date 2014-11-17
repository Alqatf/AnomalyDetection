# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import numpy as np
import nslkdd.preprocessing as preprocessing
import matplotlib
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA


if __name__ == '__main__':
    datasize = 100
    df, headers = preprocessing.get_preprocessed_data(datasize)
    print df["attack"].values.tolist()
#    binwidth = datasize / 10.0
#    histdist = plt.hist(df["duration"], binwidth, normed=True)

    rpca = RandomizedPCA(n_components=1, random_state=0)
    proj = rpca.fit_transform(df)
    print df.shape
    print proj.shape

    mi = min(proj)
    ma = max(proj)
    hist, bin_edges = np.histogram(proj, bins=np.arange(int(mi)-1,int(ma)+1), density=True)
    fig, ax = plt.subplots()
    ax.plot(range(0,len(hist)), hist)


    plt.show()

