# Train model and make predictions
import numpy as np
import pandas as pd
import h5py
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from linearDiscriminantAnalysis import mergePredictions


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def runModel(X, Y):
	X = X.values
	Y = Y.values
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)

	d = X.shape[1]

	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=d, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# Fit the model
	model.fit(X, dummy_y, nb_epoch=500, batch_size=150,  verbose=2)
	print 'Training Complete'

	return model

def saveModel(model, path):
	model.save(path)
	print 'Model saved'
	#model.save('../trained_models/ff_nn_3.h5')

def predictFromModel(model, testX, classes):
	testX = testX.values.astype(float)
	predictions = model.predict(testX)
	#print predictions
	predictions = np.argmax(predictions,axis=1)
	p = classes[predictions]

	return p


def calcAccuracy(testY, p):
	# print the accuracy
	testY = testY.values
	accuracy = sum(testY == p)/float(len(testY))
	return accuracy


# Load parameter. Set to false to train a new model and set to True to load model from existing file
def runForSubject(subject, savepath, load=False):

	dataframe = pd.read_csv("../data/training/pca_data_v2/pca_subject"+str(subject)+".csv")
	dataframe = dataframe.sample(frac=1).reset_index(drop=True)
	X = dataframe.iloc[:,:-1]
	Y = dataframe.iloc[:,-1]
	classes = np.unique(Y)

	model = 0
	if (load == False):
		print 'Training new model'
		model = runModel(X, Y)
		saveModel(model, savepath)
	else:
		print 'Using existing model'
		model = load_model(savepath)
	
	p = predictFromModel(model, X, classes)

	print 'Training accuracy = ' + str(calcAccuracy(Y, p))

	test_data = pd.read_csv('../data/testing/pca_data_v2/pca_subject'+str(subject)+'.csv')
	actual = pd.read_csv('../data/testing/ActualLables/labels_subject'+str(subject)+'_psd.csv', header=None)
	actual8 = pd.read_csv('../data/testing/ActualLables/labels8_subject'+str(subject)+'_psd.csv', header=None)
	#only get the values from the actual dataframe
	actual = actual[0]
	actual8 = actual8[0]

	p = predictFromModel(model, test_data, classes)
	p8 = mergePredictions(p, 8)
	print 'Testing Accuracy = ' + str(calcAccuracy(actual, p))
	print 'Testing Accuracy for step 8 = ' + str(calcAccuracy(actual8, p8))


if __name__ == '__main__':
	runForSubject(1, '../trained_models/feedForwardSubject1.h5', load=True)
	runForSubject(2, '../trained_models/feedForwardSubject2.h5', load=True)
	runForSubject(3, '../trained_models/feedForwardSubject3.h5', load=True)

	# Results
	# Training accuracy = 0.839190729483
	# Testing Accuracy = 0.734589041096
	# Testing Accuracy for step 8 = 0.751141552511
	# Using existing model
	# Training accuracy = 0.798653846154
	# Testing Accuracy = 0.612039170507
	# Testing Accuracy for step 8 = 0.663594470046
	# Using existing model
	# Training accuracy = 0.750583203733
	# Testing Accuracy = 0.452408256881
	# Testing Accuracy for step 8 = 0.442660550459
