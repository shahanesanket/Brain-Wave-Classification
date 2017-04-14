import pandas as pd
import numpy as np
import sys

sys.path.insert(0,'../lib/seqlearn/')

from seqlearn.hmm import MultinomialHMM
from hmmlearn.hmm import GaussianHMM


input_data = pd.read_csv('../data/train_subject1_psd01.csv', header=None)

d1 = pd.read_csv('../data/train_subject1_psd01.csv',header=None)
d2 = pd.read_csv('../data/train_subject1_psd02.csv',header=None)
d3 = pd.read_csv('../data/train_subject1_psd03.csv',header=None)

input_data = pd.concat([d1, d2, d3], axis=0)
lengths = [len(d1), len(d2), len(d3)]

clf = MultinomialHMM()
clf.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1], lengths)
pred = clf.predict(d3.iloc[:,:-1])
actual = d3.iloc[:,-1]
accuracy = sum(pred == actual)/float(len(actual))

print accuracy


