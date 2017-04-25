from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np
import pickle
from sklearn import multiclass



def trainRF(training_data, savepath, load=False):
	rf = 0
	if (load == False):
		rf = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
		#scores = cross_val_score(rf, training_data.iloc[:,:-1], training_data.iloc[:,-1], cv = 10)
		#print 'Cross Validation Accuracy = ' + str(scores.mean())
		rf.fit(training_data.iloc[:,:-1], training_data.iloc[:,-1])
		with open(savepath,'wb') as f:
			pickle.dump(rf,f)
	else:
		with open(savepath,'r') as f:
			rf = pickle.load(f)

	return rf

def trainSVC(training_data, savepath, load=False):
	svc = 0
	if (load == False):
		svc = multiclass.OneVsRestClassifier(svm.SVC(kernel='rbf'),n_jobs=-1)
		svc.fit(training_data.iloc[:,:-1], training_data.iloc[:,-1])
		with open(savepath,'wb') as f:
			pickle.dump(svc,f)
	else:
		with open(savepath,'r') as f:
			svc = pickle.load(f)

	return svc

def calcAccuracy(rf, test_data, actual):
	pred = rf.predict(test_data)
	accuracy = sum(pred == actual)/float(len(actual))
	return accuracy


def main(subject):
	# d1 = pd.read_csv('../data/training/train_subject'+str(subject)+'_psd01.csv',header=None)
	# d2 = pd.read_csv('../data/training/train_subject'+str(subject)+'_psd02.csv',header=None)
	# d3 = pd.read_csv('../data/training/train_subject'+str(subject)+'_psd03.csv',header=None)

	# input_data2 = pd.concat([d1, d2, d3], axis=0)

	input_data = pd.read_csv('../data/training/pca_data_v2/pca_subject'+str(subject)+'.csv')
	test_data = pd.read_csv('../data/testing/pca_data_v2/pca_subject'+str(subject)+'.csv')
	actual = pd.read_csv('../data/testing/ActualLables/labels_subject'+str(subject)+'_psd.csv', header=None)[0]
	rf = trainRF(input_data, '../trained_models/rf_subject'+str(subject)+'.pkl', load=True)
	svc = trainSVC(input_data, '../trained_models/svc_subject'+str(subject)+'.pkl', load=True)
	accuracy = calcAccuracy(rf, test_data, actual)
	print 'Test accuracy for Random Forest for subject '+str(subject)+' = ' + str(accuracy)
	accuracy = calcAccuracy(svc, test_data, actual)
	print 'Test accuracy SVC for subject '+str(subject)+' = ' + str(accuracy)


if __name__ == '__main__':
	main(1)
	main(2)
	main(3)

# Results - RF:
# Test accuracy = 0.731449771689
# Test accuracy = 0.608870967742
# Test accuracy = 0.477064220183

# Results - SVC
# Test accuracy = 0.731164383562
# Test accuracy = 0.561347926267
# Test accuracy = 0.495986238532

# Test accuracy for Random Forest for subject 1 = 0.731449771689
# Test accuracy SVC for subject 1 = 0.731164383562
# Test accuracy for Random Forest for subject 2 = 0.608870967742
# Test accuracy SVC for subject 2 = 0.561347926267
# Test accuracy for Random Forest for subject 3 = 0.477064220183
# Test accuracy SVC for subject 3 = 0.495986238532