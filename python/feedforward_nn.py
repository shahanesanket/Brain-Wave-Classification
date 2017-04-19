# Train model and make predictions
import numpy as np
import pandas
import h5py
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("../data/scaled_data/scaled_pca.csv")
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
dataset = dataframe.values
X = dataset[:,:-1].astype(float)
Y = dataset[:,-1]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
model.add(Dense(12, input_dim=75, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X[:], dummy_y, nb_epoch=1000, batch_size=150,  verbose=1)

# calculate predictions
predictions = model.predict(X)
print predictions
predictions = np.argmax(predictions,axis=1)
classes = np.unique(Y)
p = classes[predictions]
# print the accuracy
print sum(Y == p)/float(len(Y))

import pickle
with open('../trained_models/ff_nn_1.pkl','wb') as f:
	pickle.dump(model)