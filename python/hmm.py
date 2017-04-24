import pandas as pd
import numpy as np
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

sys.path.insert(0,'../lib/seqlearn/')

from seqlearn.hmm import MultinomialHMM
from hmmlearn.hmm import GaussianHMM


#input_data = pd.read_csv('../data/scaled_data/scaled_pca.csv')
#lengths = [len(input_data)]

# d1 = pd.read_csv('../data/training/train_subject1_psd01.csv',header=None)
# d2 = pd.read_csv('../data/training/train_subject1_psd02.csv',header=None)
# d3 = pd.read_csv('../data/training/train_subject1_psd03.csv',header=None)


d1 = pd.read_csv('../data/training/pca_data_v2/pca_subject1.csv')
#d2 = pd.read_csv('../data/training/pca_data_v2/pca_subject2.csv')
#d3 = pd.read_csv('../data/training/pca_data_v2/pca_subject3.csv')

#input_data = pd.concat([d1, d2, d3], axis=0)
input_data = d1
#lengths = [len(d1), len(d2), len(d3)]
lengths = [len(d1)]



def seqHMM():
	clf = MultinomialHMM()
	clf.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1], lengths)
	pred = clf.predict(input_data.iloc[:,:-1])
	actual = input_data.iloc[:,-1]
	accuracy = sum(pred == actual)/float(len(actual))
	print accuracy


def hmmlearnHMM():
	model = GaussianHMM(3, "full")
	model.fit(input_data.iloc[:,:-1])

	actual = input_data.iloc[:,-1]
	pred = model.predict(input_data.iloc[:,:-1])
	pred = [3 if x==2 else x for x in pred]
	pred = [2 if x==1 else x for x in pred]
	pred = [7 if x==0 else x for x in pred]

	accuracy = sum(pred == actual)/float(len(actual))


	print accuracy


if __name__ == '__main__':
	hmmlearnHMM()