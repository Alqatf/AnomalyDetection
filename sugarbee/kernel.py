import math

# Gaussian kernel similarity function
# scaling factor c = 2 sigma^2 where sigma = 1.5 * median 
# EigenCuts paper uses.
# https://spectrallyclustered.wordpress.com/2010/06/05/sprint-1-k-means-spectral-clustering/
def gaussianK(x,y,median):
    d = len(x)
    for d in len(x):

    c=2*((1.5 * median)**2)
    val = (x - y)**2 / (2*variance)
    return math.exp(val)

