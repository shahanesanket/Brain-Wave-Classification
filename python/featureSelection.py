import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer



def removeCorrelation(input_data):
	#Columns found using analysis. Check rough-sanket.py and R code for code and results
	cor = [24, 72]
	vec = list(input_data.columns)
	for c in cor:
		vec.remove(c)
	new_data = input_data[vec]
	return new_data


def runPCA(input_data, test, d):
	input_data = removeCorrelation(input_data)
	test = removeCorrelation(test)

	normZ = Normalizer()
	scaledX = normZ.fit_transform(input_data.iloc[:,:-1])
	scaledTestX = normZ.transform(test)

	pca = PCA()
	pcaX = pca.fit_transform(scaledX)
	pcaX = pd.DataFrame(pcaX)
	print 'Approx 98% variance explained by '+str(d)+' features: ' + str(pca.explained_variance_ratio_[:d].sum())

	trainY = input_data.iloc[:,-1]
	trainY = trainY.reshape(len(trainY), 1)
	trainY = pd.DataFrame(trainY)
	trainY.columns = ['Class']

	trainDataAfterPCA = pd.concat([pcaX.iloc[:,:d], trainY], axis=1)
	
	testDataAfterPCA = pca.transform(scaledTestX)
	testDataAfterPCA = pd.DataFrame(testDataAfterPCA)
	testDataAfterPCA = testDataAfterPCA.iloc[:,:d]
	
	return trainDataAfterPCA, testDataAfterPCA


def mainPCA():
	d1 = pd.read_csv('../data/Training/train_subject1_psd01.csv',header=None)
	d2 = pd.read_csv('../data/Training/train_subject1_psd02.csv',header=None)
	d3 = pd.read_csv('../data/Training/train_subject1_psd03.csv',header=None)
	input_data_s1 = pd.concat([d1, d2, d3], axis=0)
	test_s1 = pd.read_csv('../data/Testing/test_subject1_psd04.csv', header=None)

	d1 = pd.read_csv('../data/Training/train_subject2_psd01.csv',header=None)
	d2 = pd.read_csv('../data/Training/train_subject2_psd02.csv',header=None)
	d3 = pd.read_csv('../data/Training/train_subject2_psd03.csv',header=None)
	input_data_s2 = pd.concat([d1, d2, d3], axis=0)
	test_s2 = pd.read_csv('../data/Testing/test_subject2_psd04.csv', header=None)

	d1 = pd.read_csv('../data/Training/train_subject3_psd01.csv',header=None)
	d2 = pd.read_csv('../data/Training/train_subject3_psd02.csv',header=None)
	d3 = pd.read_csv('../data/Training/train_subject3_psd03.csv',header=None)
	input_data_s3 = pd.concat([d1, d2, d3], axis=0)
	test_s3 = pd.read_csv('../data/Testing/test_subject3_psd04.csv', header=None)

	train1, test1 = runPCA(input_data_s1, test_s1, 40)
	train2, test2 = runPCA(input_data_s2, test_s2, 50)
	train3, test3 = runPCA(input_data_s3, test_s3, 60)

	train1.to_csv('../data/Training/pca_data_v2/pca_subject1.csv', index=False)
	train2.to_csv('../data/Training/pca_data_v2/pca_subject2.csv', index=False)
	train3.to_csv('../data/Training/pca_data_v2/pca_subject3.csv', index=False)

	test1.to_csv('../data/Testing/pca_data_v2/pca_subject1.csv', index=False)
	test2.to_csv('../data/Testing/pca_data_v2/pca_subject2.csv', index=False)
	test3.to_csv('../data/Testing/pca_data_v2/pca_subject3.csv', index=False)

if __name__ == '__main__':
	mainPCA()