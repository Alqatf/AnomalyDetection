import pandas as pd
import math
import random
from sklearn import svm

headerfile = './nslkdd/data/kddcup.names'
datfile = './nslkdd/data/KDDTrain+_20Percent.txt'

headers=open(headerfile,'r').readlines()[1:]
headers = [h.split(':')[0] for h in headers]
headers.append('attack')
headers.append('attack_idx')


df = pd.read_csv(open(datfile), names=headers)
df.iloc[0]

# convert attack labels to integer index
attacks=open(headerfile,'r').readline().replace('.\n','').split(',')
for i in range(len(attacks)):
        df['attack'][df['attack']==attacks[i]]=i

# convert protocol_type to integer index
pt=list(set(df['protocol_type']))
for i in range(len(pt)):
        df['protocol_type'][df['protocol_type']==pt[i]]=i

# convert service to integer index
pt=list(set(df['service']))
for i in range(len(pt)):
        df['service'][df['service']==pt[i]]=i

# convert flag to integer index
pt=list(set(df['flag']))
for i in range(len(pt)):
        df['flag'][df['flag']==pt[i]]=i

# scale big nums to log2 count
scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised','num_file_creations','count','srv_count','dst_host_count', 'dst_host_srv_count']
for i in range(len(scaled)):
        for j in range(df.shape[0]):
                df[scaled[i]].iloc[j]=int(math.log(1+df[scaled[i]].iloc[j],2))

df.iloc[0]

# don't include last field, difficulty flag is meta data
for i in range(len(headers)):
        print headers[i], max(df[headers[i]])
        
df.shape # 25192, 43

# 
idx = [i for i in range(df.shape[0])]
import random
random.shuffle(idx)

ptr=20000
idxTrain=idx[0:ptr]
idxTest=idx[ptr:len(idx)]

xTrain = df.iloc[idxTrain,1:41]
yTrain = df.iloc[idxTrain,41]
xTest = df.iloc[idxTest,1:41]
yTest = df.iloc[idxTest,41]

clf = svm.SVC()
clf.fit(xTrain, yTrain) 

preds = clf.predict(xTest)

#divisor = sum(yTest==11)

tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(preds)):
    if preds[i]==11 and yTest.iloc[i] == 11:
        tp = tp + 1
    if preds[i]!=11 and yTest.iloc[i] != 11:
        tn = tn + 1
    if preds[i]!=11 and yTest.iloc[i] == 11:
        fn = fn + 1
    if preds[i]==11 and yTest.iloc[i] != 11:
        fp = fp + 1

print tp
print tn
print fp
print fn

#accuracy=clf.accuracy(
## 2804 11 in preds;  2794 11 in yTest
## TN .997
#1.0*sum([preds[i]==11 and yTest.iloc[i] == 11 for i in range(len(preds))]) / sum (yTest==11)*1.0
#
## TP .9929
#1.0*sum([preds[i]!=11 and yTest.iloc[i] != 11 for i in range(len(preds))]) / sum (yTest!=11)*1.0
#
## FN .003
#1.0*sum([preds[i]!=11 and yTest.iloc[i] == 11 for i in range(len(preds))]) / sum (yTest==11)*1.0
#
## FP .007
#1.0*sum([preds[i]==11 and yTest.iloc[i] != 11 for i in range(len(preds))]) / sum (yTest!=11)*1.0

for s in set(yTest):
        # TN .997
        attacks[s], s, sum(yTest==s), 1.*sum([preds[i]==s and yTest.iloc[i] == s for i in range(len(preds))]) / sum (yTest==s)*1.0, 1.*sum([preds[i]!=s and yTest.iloc[i] != s for i in range(len(preds))]) / sum (yTest!=s)*1.0


from sklearn.metrics import confusion_matrix
confusion_matrix(list(yTest), list(preds))

'''
array([[  39,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,  132,    0,    0,    0,    6,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,    0,    0, 1731,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,   29,    0,    0,   37,    1,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,    3,    0,    2,    8, 2761,    0,    0,    0,    0,    0,    0,    0,    3],
       [   0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,    0,    0,    0,    0,    0,    0,    5,    0,    0,    0,    0,    0,    0],
       [   0,    0,    1,    0,    6,    0,    1,    0,    0,  111,    0,    1,    0,    0,    0],
       [   0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],
       [   0,    0,    0,    0,    3,    0,    1,    0,    0,    0,    0,  144,    0,    0,    0],
       [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   99,    0,    0],
       [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   38,    0],
       [   0,    0,    0,    0,    0,    0,    4,    0,    0,    0,    0,    0,    0,    0,   22]])
'''


