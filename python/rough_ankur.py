import pandas as pd
import numpy as np
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

sys.path.insert(0,'../lib/seqlearn/')

from seqlearn.hmm import MultinomialHMM
from hmmlearn.hmm import GaussianHMM


input_data = pd.read_csv('../data/scaled_data/scaled_pca.csv')
lengths = [len(input_data)]

d1 = pd.read_csv('../data/train_subject1_psd01.csv',header=None)
d2 = pd.read_csv('../data/train_subject1_psd02.csv',header=None)
d3 = pd.read_csv('../data/train_subject1_psd03.csv',header=None)

#input_data = pd.concat([d1, d2, d3], axis=0)
lengths = [len(d1), len(d2), len(d3)]

clf = MultinomialHMM()
clf.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1], lengths)
pred = clf.predict(input_data.iloc[:,:-1])
actual = d3.iloc[:,-1]
accuracy = sum(pred == actual)/float(len(actual))
print accuracy



# Linear Discriminant Analysis

#input_data = pd.read_csv('../data/')
input_data = pd.concat([d1, d2, d3], axis=0)
model = LinearDiscriminantAnalysis()
scores = cross_val_score(model, input_data.iloc[:,:-1], input_data.iloc[:,-1], cv = 10)
print scores.mean()

#model.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1])
pred = model.predict(d3.iloc[:,:-1])
actual = d3.iloc[:,-1]
accuracy = sum(pred == actual)/float(len(actual))
print accuracy


d1 = pd.read_csv('../data/train_subject2_psd01.csv',header=None)
d2 = pd.read_csv('../data/train_subject2_psd02.csv',header=None)
d3 = pd.read_csv('../data/train_subject2_psd03.csv',header=None)

d1 = pd.read_csv('../data/train_subject3_psd01.csv',header=None)
d2 = pd.read_csv('../data/train_subject3_psd02.csv',header=None)
d3 = pd.read_csv('../data/train_subject3_psd03.csv',header=None)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np
import pickle
from sklearn import multiclass


input_data = pd.read_csv('../data/scaled_data/scaled_pca.csv')
input_data2 = pd.concat([d1, d2, d3], axis=0)
training_data = input_data
rf = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
#rf.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1])
scores = cross_val_score(rf, training_data.iloc[:,:-1], training_data.iloc[:,-1], cv = 10)
print scores
# array([ 0.61895613,  0.66122318,  0.70637208,  0.56644252,  0.5429212 , 0.52946829,  0.54615385,  0.44038462,  0.44807692,  0.39807692])
# 10 trees 51, 20 trees 52, 40 trees 55, 100 trees 56
svc = multiclass.OneVsRestClassifier(svm.SVC(kernel='rbf'),n_jobs=-1)
#svc.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1])
scores = cross_val_score(svc, training_data.iloc[:,:-1], training_data.iloc[:,-1], cv = 10, verbose=5)

actual = input_data.iloc[:,-1]
pred = svc2.predict(input_data.iloc[:,:-1])
accuracy = sum(pred == actual)/float(len(actual))
print accuracy


with open('../trained_models/rf_all_subjects.pkl','wb') as f:
	pickle.dump(rf,f)
with open('../trained_models/svc_all_subject.pkl','wb') as f:
	pickle.dump(svc,f)
with open('../trained_models/rf_results.txt','w') as f:
    for item in scores:
        f.write("%s\n" % item)
print scores





#Processing test data
#Removing correlation

test1 = pd.read_csv('../data/Testing/test_subject1_psd04.csv', header=None)
test2 = pd.read_csv('../data/Testing/test_subject2_psd04.csv', header=None)
test3 = pd.read_csv('../data/Testing/test_subject3_psd04.csv', header=None)

correlated_features = [24,72]

vec = list(test1.columns)
for v in correlated_features:
	vec.remove(v)

test1_uncor = test1[vec]
test2_uncor = test2[vec]
test3_uncor = test3[vec]

test1_uncor.to_csv('../data/Testing/uncorrelated_subject1_data.csv',index=False)
test2_uncor.to_csv('../data/Testing/uncorrelated_subject2_data.csv',index=False)
test3_uncor.to_csv('../data/Testing/uncorrelated_subject3_data.csv',index=False)


# PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing

train = pd.read_csv('../data/Training/uncorrelated_training_data.csv')
vec = list(train.columns)
train = train[vec[:-1]]
#preprocessing.scale(temp,axis=1,copy=False)
pca = PCA()
pca.fit(train)

test1_pca = pca.transform(test1_uncor)
test2_pca = pca.transform(test2_uncor)
test3_pca = pca.transform(test3_uncor)

test1_pca = pd.DataFrame(test1_pca)
test2_pca = pd.DataFrame(test2_pca)
test3_pca = pd.DataFrame(test3_pca)

test1_pca.iloc[:,:40].to_csv('../Data/Testing/pca_subject1.csv', index=False)
test2_pca.iloc[:,:40].to_csv('../Data/Testing/pca_subject2.csv', index=False)
test3_pca.iloc[:,:40].to_csv('../Data/Testing/pca_subject3.csv', index=False)


