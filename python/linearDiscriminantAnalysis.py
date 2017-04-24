import pandas as pd
import numpy as np
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from scipy.stats import mode


def LDA(input_data, test, actual, actual8):
	model = LinearDiscriminantAnalysis()
	scores = cross_val_score(model, input_data.iloc[:,:-1], input_data.iloc[:,-1], cv = 10)
	print 'Cross Validation Accuracy = ' + str(scores.mean())
	model.fit(input_data.iloc[:,:-1], input_data.iloc[:,-1])
	pred = model.predict(test)
	accuracy = sum(pred == actual)/float(len(actual))
	pred8 = mergePredictions(pred, 8)
	accuracy8 = sum(pred8 == actual8)/float(len(actual8))
	print 'Test Accuracy for the subject is = ' + str(accuracy)
	print 'Test Accuracy for the subject at step 8 is = ' + str(accuracy8)


def mergePredictions(predictions, step=8):
	L = len(predictions)/step
	output = []
	for i in range(L):
		m = mode(predictions[range(i*step, (i+1)*step)])[0][0]
		output.append(m)
	return output


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

	LDA(input_data_s1, test_s1, actual_s1, actual8_s1)
	LDA(input_data_s2, test_s2, actual_s2, actual8_s2)
	LDA(input_data_s3, test_s3, actual_s3, actual8_s3)

	# Results:
	# Cross Validation Accuracy = 0.69484335286
	# Test Accuracy for the subject is = 0.714611872146
	# Test Accuracy for the subject at step 8 is = 0.728310502283
	# Cross Validation Accuracy = 0.539059371463
	# Test Accuracy for the subject is = 0.581221198157
	# Test Accuracy for the subject at step 8 is = 0.582949308756
	# Cross Validation Accuracy = 0.461690546146
	# Test Accuracy for the subject is = 0.491685779817
	# Test Accuracy for the subject at step 8 is = 0.502293577982

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

	LDA(input_data_s1, test_s1, actual_s1, actual8_s1)
	LDA(input_data_s2, test_s2, actual_s2, actual8_s2)
	LDA(input_data_s3, test_s3, actual_s3, actual8_s3)

	# Results:
	# Cross Validation Accuracy = 0.719154247244
	# Test Accuracy for the subject is = 0.732591324201
	# Test Accuracy for the subject at step 8 is = 0.748858447489
	# Cross Validation Accuracy = 0.565501871535
	# Test Accuracy for the subject is = 0.613479262673
	# Test Accuracy for the subject at step 8 is = 0.624423963134
	# Cross Validation Accuracy = 0.482587359569
	# Test Accuracy for the subject is = 0.505447247706
	# Test Accuracy for the subject at step 8 is = 0.504587155963

if __name__ == '__main__':
	mainPCAData()