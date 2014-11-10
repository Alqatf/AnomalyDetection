import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.pyplot
import matplotlib.mlab

def discrete_by_gmm(df, key, n_components):
    clf = mixture.GMM(n_components=2, covariance_type='full', min_covar=0.00001)
    clf.fit(df[key])

    m1, m2 = clf.means_
    w1, w2 = clf.weights_
    c1, c2 = clf.covars_
    
    # show hist
    histdist = plt.hist(df[key], 100, normed=True, alpha=0.2)
    
    # show gausses
    plotgauss1 = lambda x: plt.plot(x,w1*matplotlib.mlab.normpdf(x,m1,np.sqrt(c1))[0], linewidth=2, color='k')
    plotgauss2 = lambda x: plt.plot(x,w2*matplotlib.mlab.normpdf(x,m2,np.sqrt(c2))[0], linewidth=2, color='r')
    plotgauss1(histdist[1])
    plotgauss2(histdist[1])


    plt.show()

    return df
#
#    for i, _ in enumerate(l):
#        df[key][df[key]==l[i]]=i
#    return df
#
#
#    ret = []
#    for i in data:
#        d = clf.predict(data)
#        ret.append(d)
#    return ret

if __name__ == '__main__':
    from data import model
    headerfile = './data/kddcup.names'
    datafile = './data/KDDTrain+_20Percent.txt'

    headers, attacks = model.load_headers(headerfile)
    df = model.load_dataframe(datafile,headers,datasize=100)
    df = discrete_by_gmm(df, "dst_bytes", 2)

#    df[targets[i]].iloc[j]=int(math.log(1+df[targets[i]].iloc[j],2))

