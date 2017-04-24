import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
from linearDiscriminantAnalysis import mergePredictions

sys.path.insert(0,'../lib/seqlearn/')

from seqlearn.perceptron import StructuredPerceptron



def perceptronTest(input_data, test, actual, actual8):
	model = StructuredPerceptron(verbose=False, random_state=37, max_iter=1000)

	l = 9*len(input_data)/10 - 1
	scores = cross_val_score(model, input_data.iloc[:,:-1], input_data.iloc[:,-1], cv = 10, fit_params={'lengths':[l]})
	
	print 'Cross Validation Accuracy = ' + str(scores.mean())
	
	model.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1], [len(input_data)])
	pred = model.predict(test)
	accuracy = sum(pred == actual)/float(len(actual))
	pred8 = mergePredictions(pred, 8)
	accuracy8 = sum(pred8 == actual8)/float(len(actual8))
	print 'Test Accuracy for the subject is = ' + str(accuracy)
	print 'Test Accuracy for the subject at step 8 is = ' + str(accuracy8)


def mainRawData():
	#Loading Input data (training)
	d1 = pd.read_csv('../data/training/train_subject1_psd01.csv',header=None)
	d2 = pd.read_csv('../data/training/train_subject1_psd02.csv',header=None)
	d3 = pd.read_csv('../data/training/train_subject1_psd03.csv',header=None)
	input_data_s1 = pd.concat([d1, d2, d3], axis=0)

	d1 = pd.read_csv('../data/training/train_subject2_psd01.csv',header=None)
	d2 = pd.read_csv('../data/training/train_subject2_psd02.csv',header=None)
	d3 = pd.read_csv('../data/training/train_subject2_psd03.csv',header=None)
	input_data_s2 = pd.concat([d1, d2, d3], axis=0)

	d1 = pd.read_csv('../data/training/train_subject3_psd01.csv',header=None)
	d2 = pd.read_csv('../data/training/train_subject3_psd02.csv',header=None)
	d3 = pd.read_csv('../data/training/train_subject3_psd03.csv',header=None)
	input_data_s3 = pd.concat([d1, d2, d3], axis=0)

	#Loading Test data (all subjects)
	test_s1 = pd.read_csv('../data/testing/test_subject1_psd04.csv', header=None)
	actual_s1 = pd.read_csv('../data/testing/ActualLables/labels_subject1_psd.csv', header=None)
	actual8_s1 = pd.read_csv('../data/testing/ActualLables/labels8_subject1_psd.csv', header=None)
	actual_s1 = actual_s1[0]
	actual8_s1 = actual8_s1[0]

	test_s2 = pd.read_csv('../data/testing/test_subject2_psd04.csv', header=None)
	actual_s2 = pd.read_csv('../data/testing/ActualLables/labels_subject2_psd.csv', header=None)
	actual8_s2 = pd.read_csv('../data/testing/ActualLables/labels8_subject2_psd.csv', header=None)
	actual_s2 = actual_s2[0]
	actual8_s2 = actual8_s2[0]

	test_s3 = pd.read_csv('../data/testing/test_subject3_psd04.csv', header=None)
	actual_s3 = pd.read_csv('../data/testing/ActualLables/labels_subject3_psd.csv', header=None)
	actual8_s3 = pd.read_csv('../data/testing/ActualLables/labels8_subject3_psd.csv', header=None)
	actual_s3 = actual_s3[0]
	actual8_s3 = actual8_s3[0]

	perceptronTest(input_data_s1, test_s1, actual_s1, actual8_s1)
	perceptronTest(input_data_s2, test_s2, actual_s2, actual8_s2)
	perceptronTest(input_data_s3, test_s3, actual_s3, actual8_s3)

	# Results:
	# Cross Validation Accuracy = 0.71953708592
	# Test Accuracy for the subject is = 0.768835616438
	# Test Accuracy for the subject at step 8 is = 0.762557077626
	# Cross Validation Accuracy = 0.571821113391
	# Test Accuracy for the subject is = 0.616359447005
	# Test Accuracy for the subject at step 8 is = 0.612903225806
	# Cross Validation Accuracy = 0.383837959864
	# Test Accuracy for the subject is = 0.406823394495
	# Test Accuracy for the subject at step 8 is = 0.389908256881

def mainPCAData():
	input_data_s1 = pd.read_csv('../data/training/pca_data_v2/pca_subject1.csv')

	input_data_s2 = pd.read_csv('../data/training/pca_data_v2/pca_subject2.csv')

	input_data_s3 = pd.read_csv('../data/training/pca_data_v2/pca_subject3.csv')

	#Loading Test data (all subjects)
	test_s1 = pd.read_csv('../data/testing/pca_data_v2/pca_subject1.csv')
	actual_s1 = pd.read_csv('../data/testing/ActualLables/labels_subject1_psd.csv', header=None)
	actual8_s1 = pd.read_csv('../data/testing/ActualLables/labels8_subject1_psd.csv', header=None)
	actual_s1 = actual_s1[0]
	actual8_s1 = actual8_s1[0]

	test_s2 = pd.read_csv('../data/testing/pca_data_v2/pca_subject2.csv')
	actual_s2 = pd.read_csv('../data/testing/ActualLables/labels_subject2_psd.csv', header=None)
	actual8_s2 = pd.read_csv('../data/testing/ActualLables/labels8_subject2_psd.csv', header=None)
	actual_s2 = actual_s2[0]
	actual8_s2 = actual8_s2[0]

	test_s3 = pd.read_csv('../data/testing/pca_data_v2/pca_subject3.csv')
	actual_s3 = pd.read_csv('../data/testing/ActualLables/labels_subject3_psd.csv', header=None)
	actual8_s3 = pd.read_csv('../data/testing/ActualLables/labels8_subject3_psd.csv', header=None)
	actual_s3 = actual_s3[0]
	actual8_s3 = actual8_s3[0]

	perceptronTest(input_data_s1, test_s1, actual_s1, actual8_s1)
	perceptronTest(input_data_s2, test_s2, actual_s2, actual8_s2)
	perceptronTest(input_data_s3, test_s3, actual_s3, actual8_s3)

	# Results:
	# Cross Validation Accuracy = 0.870042431112
	# Test Accuracy for the subject is = 0.882705479452
	# Test Accuracy for the subject at step 8 is = 0.876712328767
	# Cross Validation Accuracy = 0.754951838489
	# Test Accuracy for the subject is = 0.802131336406
	# Test Accuracy for the subject at step 8 is = 0.797235023041
	# Cross Validation Accuracy = 0.523014108367
	# Test Accuracy for the subject is = 0.579128440367
	# Test Accuracy for the subject at step 8 is = 0.584862385321

if __name__ == '__main__':
	mainRawData()