# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""
print (__doc__)

import cPickle as pickle
import nslkdd.preprocessing as preprocessing

if __name__ == '__main__':
    import time
    start = time.time()
    datasize = 1000

    print "preprocessing data..."
    df, headers = preprocessing.get_preprocessed_data(datasize)
    with open('df.pkl','wb') as output:
        pickle.dump(df, output,-1)
    with open('headers.pkl','wb') as output:
        pickle.dump(headers, output,-1)

    elapsed = (time.time() - start)
    print "done in %s seconds" % (elapsed)
