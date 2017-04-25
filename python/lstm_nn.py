import numpy as np
import pandas as pd
import h5py
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

time_step = 8

def generateDataFormat1(X):
	X = X.values
	print X.shape
	#dataX = np.zeros(shape=(X.shape[0]-time_step-1,time_step,X.shape[1]))
	dataX = np.zeros(shape=(10520,8,40))
	print dataX.shape
	for i in range(0,dataX.shape[0]):
		dataX[i] = X[i:i+time_step]
	return dataX

def generateLabelFormat1(Y):
	print Y.shape
	dataY = Y[8:]
	print dataY.shape
	return dataY

dataframe = pd.read_csv("../data/training/pca_data_v2/pca_subject1.csv")
#dataframe = dataframe.sample(frac=1).reset_index(drop=True)
X = dataframe.iloc[:,:-1]
Y = dataframe.iloc[:,-1]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print dummy_y
dataX = generateDataFormat1(X)	#8 data instances for one data classification
dataY = generateLabelFormat1(dummy_y) # encoded labels starting 8th sample
# now need to shuffle dataX and dataY with the same shuffling vector.
# generate random integers of dataX.shape[0] and store in a list variable and use as index to shuffle data with windows
shuffle_vector = np.random.randint(low=0, high=dataX.shape[0], size=dataX.shape[0])
dataX=dataX[shuffle_vector]
dataY=dataY[shuffle_vector]
# now design the LSTM model and while fiting set shuffle=False
model = Sequential()
model.add(LSTM(32,input_shape=(dataX.shape[1],dataX.shape[2])))
model.add(Dense(dataY.shape[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(dataX, dataY, epochs=50, batch_size=11, verbose=2, shuffle=False, validation_split=0.1)
# save the trained model
path = '../trained_models/feedForwardSubject1_LSTM2.h5'
model.save(path)
print 'model trained and saved!'