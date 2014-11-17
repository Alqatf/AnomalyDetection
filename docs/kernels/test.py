import math
# 
def linearK(x, y):
    s = 1;
    for i in range(2):
        s+= (x[i] * y[i])
    return s

def polyK(x,y,p=2):
    return pow(linearK(x,y),p)

def radialK(x,y,sigma=12):
    diff = range(2)
    for i in range(2):
        diff[i] = x[i] - y[i]
    val = 0
    for i in range(2):
        val += diff[i] * diff[i]
    return math.exp(-(val) / (2 * sigma * sigma))

def sigmoidK(x,y,k=.005,delta=-.03):
    val = 0;
    for i in range(2):
        val += x[i]*y[i]
    val *= k
    val -= delta
    return math.tanh(val)

# Gaussian kernel similarity function
# scaling factor c = 2 sigma^2 where sigma = 1.5 * median 
# EigenCuts paper uses.
# https://spectrallyclustered.wordpress.com/2010/06/05/sprint-1-k-means-spectral-clustering/
def gaussianK(x,y,median):
    c=2*((1.5 * median)**2)
    val = (x - y)**2 / (2*variance)
    return math.exp(val)



