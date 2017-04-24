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
from scipy.stats import mode
from sklearn.model_selection import cross_val_score

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
	print dummy_y
	d = X.shape[1]
	# create model
	model = Sequential()
	model.add(Dense(50, input_dim=d, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(8, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	
	#scores = cross_val_score(model, X, dummy_y, cv = 10, fit_params={'nb_epoch':500, 'batch_size':200, 'verbose':0})
	
	#print 'Cross Validation Accuracy = ' + str(scores.mean())
	model.fit(X, dummy_y, nb_epoch=500, batch_size=200, verbose=2)
	print 'Training Complete'
	return model

def saveModel(model, path):
	model.save(path)
	print 'Model saved'
	#model.save('../trained_models/ff_nn_3.h5')

def predictFromModel(model, testX, testY, classes):
	testX = testX.values.astype(float)
	testY = testY.values
	predictions = model.predict(testX)
	predictions = np.argmax(predictions,axis=1)
	p = classes[predictions]
	testY = testY.reshape(1,-1)[0]
	# print the accuracy
	accuracy = sum(testY == p)/float(len(testY))
	print accuracy
	return accuracy

def mergeEightRows(X):
	mainFrame = pd.DataFrame()
	for i in range(0,len(X),8):
		if i+7<len(X):
			currRow=pd.DataFrame(X.iloc[i:i+8,:].as_matrix().flatten().reshape(1,320))
			mainFrame = pd.concat([mainFrame,currRow],axis=0)
	return mainFrame

def mergePredictions(predictions, step=8):
	L = len(predictions)/step
	output = []
	for i in range(L):
		m = mode(predictions[range(i*step, (i+1)*step)])[0][0]
		output.append(m)
	return pd.DataFrame(output)

# Load parameter. Set to false to train a new model and set to True to load model from existing file
def runForSubject(subject, savepath, load=False):
	subject=1
	dataframe = pd.read_csv("../data/training/pca_data_v2/pca_subject"+str(subject)+".csv")
	X = mergeEightRows(dataframe.iloc[:,:-1])
	Y =mergePredictions(dataframe.iloc[:,-1])
	df = X
	df['Class'] = Y.values
	df = df.sample(frac=1).reset_index(drop=True)
	X = df.iloc[:,:-1]
	Y = df.iloc[:,-1]
	classes = np.unique(Y)
	model = 0
	if (load == False):
		print 'Training new model'
		model = runModel(X, Y)
		saveModel(model, savepath)
	else:
		print 'Using existing model'
		model = load_model(savepath)
	print 'Training accuracy = ' + str(predictFromModel(model, X, Y, classes))
	test_data = pd.read_csv('../data/testing/pca_data_v2/pca_subject'+str(subject)+'.csv')
	test_data = mergeEightRows(test_data)
	actual = pd.read_csv('../data/testing/ActualLables/labels8_subject'+str(subject)+'_psd.csv', header=None)
	#only get the values from the actual dataframe
	actual = actual[0]
	print 'Testing Accuracy = ' + str(predictFromModel(model, test_data, actual, classes))




if __name__ == '__main__':
	runForSubject(1, '../trained_models/feedForwardSubject1_8.h5', load=False)
	#runForSubject(2, '../trained_models/feedForwardSubject2_8.h5', load=False)
	#runForSubject(3, '../trained_models/feedForwardSubject3_8.h5', load=True)

	# Results
	# Training accuracy = 0.839190729483
	# Testing Accuracy = 0.734589041096
	# Using existing model
	# Training accuracy = 0.798653846154
	# Testing Accuracy = 0.612039170507
	# Using existing model
	# Training accuracy = 0.750583203733
	# Testing Accuracy = 0.452408256881

#scratch:

from keras.layers import LSTM
dataframe = pd.read_csv("../data/training/pca_data_v2/pca_subject1.csv")
#dataframe = dataframe.sample(frac=1).reset_index(drop=True)
X = dataframe.iloc[:,:-1]
Y = dataframe.iloc[:,-1]

X = X.values
Y = Y.values
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print dummy_y
X=X.reshape(1316*8,1,40)
dummy_y=dummy_y.reshape(1316*8,1,3)

model=Sequential()
model.add(LSTM(12,input_shape=(1,40),return_sequences=True))
#model.add(Dense(8, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, nb_epoch=400, batch_size=200, verbose=2)
path = '../trained_models/feedForwardSubject1_LSTM1.h5'
model.save(path)


dataframe = pd.read_csv('../data/testing/pca_data_v2/pca_subject1.csv')
actual = pd.read_csv('../data/testing/ActualLables/labels_subject1_psd.csv', header=None)
X = dataframe.values
Y = actual[0].values
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print dummy_y
X=X.reshape(-1,1,40)
dummy_y=dummy_y.reshape(-1,1,3)

predictions = model.predict(X)
predictions = predictions.reshape(3504,3)
predictions = np.argmax(predictions,axis=1)
classes = np.unique(Y)
p = classes[predictions]
accuracy = sum(Y == p)/float(len(Y))

# LSTM test subject 1: 0.71518264840182644, Train: 0.88
