'''
Builds a model to distinguish between correct and incorrect combinations of decay products
Performs binary classification using a Deep Neural Network
Takes a csv of training data and an identifying outStr. Outputs a .h5 model and plots of the performance
Usage: 
   python3.6 match_keras.py <input csv file> <outStr>
'''

#import relevant modules
import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # feed-forward neural network (sequential layers)                         
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization # fully interconnected layers  
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#import keras
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy
from matchPlots import make_plots

tf.keras.backend.set_floatx('float32') #reduce tensorflow precision, runs faster

#load in command line arguments
inFile = sys.argv[1]
outDir = sys.argv[2]

#Use optimal parameters obtained from grid search
if outDir=='top2lSS' or outDir=='all2lSS':
    best_params = {"epochs": 80, "layers": 6, "nodes": 60}
elif outDir=='top3l' or outDir=='all3l':
    #best_params = {'epochs': 40, 'layers': 6, 'nodes': 50}
    best_params = {'epochs': 80, 'layers': 6, 'nodes': 70}
#elif outDir=='higgsTop2lSS':
elif 'higgsTop2lSS' in outDir: 
    best_params = {'epochs': 80, 'layers': 7, 'nodes': 60}
elif outDir=='higgsTop3lS':
    best_params = {'epochs': 100, 'layers': 7, 'nodes': 70}
elif outDir=='higgsTop3lF':
    best_params = {'epochs': 120, 'layers': 5, 'nodes': 60}
else:
    best_params = {"epochs": 80, "layers": 5, "nodes": 50}
    #best_params = {"epochs": 120, "layers": 6, "nodes": 90}

print(best_params)

#load in the training data
inDF = pd.read_csv(inFile, index_col=False)
inDF = sk.utils.shuffle(inDF) #shuffle data 

#normalize input data
maxVals = inDF.max()
minVals = inDF.min()
inDF=(inDF-inDF.min())/(inDF.max()-inDF.min())

#save normalizations for future use
normFactors = [maxVals.drop(['match']), minVals.drop(['match'])]
#normFactors = np.asarray(maxVals.drop(['match']))
np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1 # track the dimention of the input
print(list(inDF))
train, test = train_test_split(inDF, test_size=0.1) #split to train and test. Large train set means 10% is good enough
 
#separate the labels from the training set
y_train = train['match']
y_test = test['match']

train = train.drop(['match'],axis=1)
test = test.drop(['match'],axis=1)

test, train = test.values, train.values

def create_model(layers=best_params['layers'], nodes=best_params['nodes'], activation='LeakyReLU', regularizer=None):
    '''
    builds a keras models using the hyperparameters provided. Layers are fully connected, and Adam is used for the optimizer
    '''

    model = Sequential()
    model.add(Dense(nodes, input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    # hidden layer: 5 nodes by default
    for l in range(layers):
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        model.add(Dense(nodes, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(1, activation=activation))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['AUC'])
    return model

#model=KerasClassifier(build_fn=create_model, verbose=1)
model=KerasRegressor(build_fn=create_model, verbose=1) #build the model
result=model.fit(train, y_train, validation_split=0.1, epochs=best_params['epochs']) # train the model

model.model.save("models/keras_model_"+outDir+".h5") # save the output

#plot the performance of the mode
y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

make_plots('keras', result, outDir, y_train, y_test, y_train_pred, y_test_pred)
