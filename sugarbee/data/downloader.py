import urllib

d1 = "KDDTrain+.arff"
d2 = "KDDTrain+.txt"
d3 = "KDDTrain+_20Percent.arff"
d4 = "KDDTrain+_20Percent.txt"
d5 = "KDDTest+.arff"
d6 = "KDDTest+.txt"
d7 = "KDDTest-21.arff"
d8 = "KDDTest-21.txt"

datasets = [d1,d2,d3,d4,d5,d6,d7,d8]
source_path = "http://nsl.cs.unb.ca/NSL-KDD/" 

for dataset in datasets:
    source_name = dataset
    source = source_path + source_name
    target = source_name
    print "downloading... " + target
    urllib.urlretrieve(source, target)

