'''                                                                                                                         
Builds a model to distinguish between correct and incorrect combinations of decay products                             
Performs regressing of the Higgs Pt spectrum using a Deep Neural Network                                                  
Takes a csv of training data and an identifying outStr. Outputs a .h5 model and plots of the performance               \
Usage:                                                                                                                   
python3.6 pt_keras.py <input csv file> <outStr>                                                                      
'''

#load relevant modules
import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import tensorflow.keras
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy
from pt_plots import makePlots

inFile = sys.argv[1]
outDir = sys.argv[2]

##Use optimal parameters obtained from grid search 
if '2lSS' in outDir:
    epochs = 150
    layers = 6
    nodes = 75
elif outDir=='higgs3lS' or outDir=='higgsTop3lS' or outDir=='testHiggsTop3lS':
    epochs = 150
    layers = 6
    nodes = 75
elif outDir=='higgs3lF' or outDir=='higgsTop3lF':
    epochs = 150
    layers = 5
    nodes = 60
else:
    epochs = 120
    layers = 5
    nodes=50

#load in the training data 
inDF = pd.read_csv(inFile, index_col=False)
inDF = sk.utils.shuffle(inDF)

#normalize input data. save norm parameters for future use of the model
maxVals = inDF.max()
minVals = inDF.min()
yMax = inDF['higgs_pt'].max() 
inDF = (inDF-minVals)/(maxVals-minVals)
normFactors = [maxVals.drop(['higgs_pt']), minVals.drop(['higgs_pt']), yMax]
np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

#use 10% test split
train, test = train_test_split(inDF, test_size=0.1)

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

test, train = test.values, train.values

def create_model(layers=layers, nodes=nodes, regularizer=None, activation='relu'):
    '''                                                                                                                     
    builds a keras models using the hyperparameters provided. Layers are fully connected, and Adam is used for the optimizer
    '''  

    from keras.models import Sequential # feed-forward neural network (sequential layers)
    from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization  # fully interconnected layers
    model = Sequential()
    #model.add(Dense(layers[0], input_dim = nFeatures, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(nodes, input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    #for l in layers:
    for l in range(layers):
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        #model.add(Dropout(0.2))
        model.add(Dense(nodes, activation=activation, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
        #model.add(BatchNormalization())
    # one output, mapped to [0,1] by sigmoid function
    model.add(Dense(1, activation='sigmoid'))
    # assemble the model (Translate to TensorFlow)
    model.compile(loss="mean_squared_error", optimizer='adam')
    #model.compile(loss="mean_absolute_error", optimizer='adam')
    return model

#load up the model
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, epochs=epochs) # fit the model to data

model.model.save("models/keras_model_"+outDir+".h5") # save the model

y_pred_train = model.predict(train) # run model predictions on test and training set
y_pred_test = model.predict(test)

#rescale outputs
y_train = y_train*yMax
y_test = y_test*yMax
y_train_pred = y_pred_train*yMax
y_test_pred = y_pred_test*yMax

if '2lSS' in outDir:
    y_train_pred = 1.3*(y_train_pred-30e3)
    y_test_pred = 1.3*(y_test_pred-30e3)
elif '3lS' in outDir:
    y_train_pred = 1.2*(y_train_pred-20e3)                                                                             
    y_test_pred = 1.2*(y_test_pred-20e3)

#plot the performance of the model
makePlots('keras', result, outDir, y_train, y_test, y_train_pred, y_test_pred)
