from data import model
import preprocessing

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

def main():
    headerfile = './data/kddcup.names'
    datafile = './data/KDDTrain+_20Percent.txt'

    headers, attacks = model.load_headers(headerfile)
    df = model.load_dataframe(datafile,headers,datasize=100)

    protocols = list(set(df['protocol_type']))
    services = list(set(df['service']))
    flags = list(set(df['flag']))

    df = preprocessing.discretize_elems_in_list(df,attacks,'attack')
    df = preprocessing.discretize_elems_in_list(df,protocols,'protocol_type')
    df = preprocessing.discretize_elems_in_list(df,services,'service')
    df = preprocessing.discretize_elems_in_list(df,flags,'flag')

    scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised','num_file_creations','count','srv_count','dst_host_count', 'dst_host_srv_count']

    df = preprocessing.scale_bignums_to_log2(df,scaled)

    print "preprocessing finished"

    key = 'service'

    target = df[key]
    print len(target)
    skf = StratifiedKFold(target, n_folds=4)
    train_index, test_index = next(iter(skf))

#    X_train = df[key][train_index]
#    X_test = df[key][test_index]
#    n_classes = len(np.unique(X_train))
#
#    print X_train
#    print len(X_train)
#    classifiers = dict((covar_type, GMM(n_components=n_classes,
#                        covariance_type=covar_type, init_params='wc', n_iter=20))
#                       for covar_type in ['spherical', 'diag', 'tied', 'full'])
#
#
#    n_classifiers = len(classifiers)
#
#    plt.figure(figsize=(3 * n_classifiers / 2, 6))
#    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
#                        left=.01, right=.99)
#
#
#    for index, (name, classifier) in enumerate(classifiers.items()):
#        print index
#        print name
#        # Since we have class labels for the training data, we can
#        # initialize the GMM parameters in a supervised manner.
#        classifier.means_ = np.array([X_train.mean(axis=0)
#                                      for i in xrange(n_classes)])
#        print classifier.means_
##    
##        # Train the other parameters using the EM algorithm.
##        classifier.fit(X_train)
##    
##        h = plt.subplot(2, n_classifiers / 2, index + 1)
##        make_ellipses(classifier, h)
##    
##        for n, color in enumerate('rgb'):
##            data = iris.data[iris.target == n]
##            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
##                        label=iris.target_names[n])
##        # Plot the test data with crosses
##        for n, color in enumerate('rgb'):
##            data = X_test[y_test == n]
##            plt.plot(data[:, 0], data[:, 1], 'x', color=color)
##    
##        y_train_pred = classifier.predict(X_train)
##        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
##        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
##                 transform=h.transAxes)
##    
##        y_test_pred = classifier.predict(X_test)
##        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
##        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
##                 transform=h.transAxes)
##    
##        plt.xticks(())
##        plt.yticks(())
##        plt.title(name)
#    
#    plt.legend(loc='lower right', prop=dict(size=12))
#    
#    
#    plt.show()




if __name__ == '__main__':
    main()

