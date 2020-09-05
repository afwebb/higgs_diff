import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy
from pt_plots import makePlots

inFile = sys.argv[1]
outDir = sys.argv[2]

if outDir=='higgsTop2lSS' or outDir=='higgsTop2lSSBin':
    epochs = 120
    layers = 8
    nodes = 75
elif outDir=='higgsTop3lF' or outDir=='higgsTop3lFBin':
    epochs = 120
    layers = 8
    nodes = 75
elif outDir=='higgsTop3lS' or outDir=='higgsTop3lSBin':
    epochs = 120
    layers = 8
    nodes = 75
else:
    epochs = 6
    layers = 6
    nodes = 75

inDF = pd.read_csv(inFile, index_col=False)
#inDF['higgs_pt'] = pd.cut(inDF['higgs_pt'], bins=[0, 150000, 9999999999], labels=[0,1])
inDF['higgs_pt'][inDF.higgs_pt<150000] = 0
inDF['higgs_pt'][inDF.higgs_pt>150000] = 1

inDF = sk.utils.shuffle(inDF)
maxVals = inDF.max()
minVals = inDF.min()
#inDF = (inDF-minVals)/(maxVals-minVals)

normFactors = [maxVals.drop(['higgs_pt']), minVals.drop(['higgs_pt'])]
np.save('models/bin_'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

train, test = train_test_split(inDF, test_size=0.1)

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

train = (train-minVals.drop(['higgs_pt']))/(maxVals.drop(['higgs_pt']) - minVals.drop(['higgs_pt']) )
test = (test-minVals.drop(['higgs_pt']))/(maxVals.drop(['higgs_pt']) - minVals.drop(['higgs_pt']) ) 

test, train = test.values, train.values
#train = tf.convert_to_tensor(train.values)
#test = tf.convert_to_tensor(test.values)

def create_model(layers=layers, nodes=nodes, regularizer=None, activation='relu'):
    from keras.models import Sequential # feed-forward neural network (sequential layers)
    from keras.layers import Dense, Dropout, LeakyReLU  # fully interconnected layers
    model = Sequential()
    #model.add(Dense(layers[0], input_dim = nFeatures, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(nodes, input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    for l in range(layers):
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        #model.add(Dropout(0.2))
        model.add(Dense(nodes, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
    # one output, mapped to [0,1] by sigmoid function
    model.add(Dense(1, activation='sigmoid'))
    # assemble the model (Translate to TensorFlow)
    #model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['AUC'])
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, epochs=epochs)

model.model.save("models/bin_keras_model_"+outDir+".h5")

y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

makePlots('bin_keras', model, outDir, y_train, y_test, y_train_pred, y_test_pred)
