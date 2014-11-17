# -*- coding: utf-8 -*-
"""
===========================================
Realisation plotting for data check
===========================================

In order to understand the data more, it will visualise 20% dataset for every properties.

It usually takes about 250 seconds
"""
print (__doc__)

#import sugarbee.affinity as affinity
import nslkdd.preprocessing as preprocessing


# Spectral clustering
def sc_run():
    datasize = 2
    df, headers = preprocessing.get_preprocessed_data(datasize)
    print df.head(1)

#    kernels = {"gaussian"}
#    distances = {"L1"}
#    print "==========="
#    print df.values
#    print df.index
#    print df.columns
#    print "==========="
#    
#    affinity_matrix = affinity.get_affinity_matrix(df, headers, kernel, distance)

    #for k in kernels:
    #    for d in distances:
    #        affinity_matrix = sugarbee.get_affinity_matrix(df, headers, k, d)
    #        cluster_model = sugerbee.clustering(affinity_matrix)
    #        sugarbee.validation(cluster_model)

# Random forest
def rf_run():
    datasize = 2
    df, headers = preprocessing.get_preprocessed_data(datasize)
    pass

if __name__ == '__main__':
    sc_run()


