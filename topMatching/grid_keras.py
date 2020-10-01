import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy
import json
tf.keras.backend.set_floatx('float32')

inFile = sys.argv[1]
outDir = sys.argv[2]

resultFile = open('models/gridResults_'+outDir+'.txt', 'w')

inDF = pd.read_csv(inFile, index_col=False)
inDF = sk.utils.shuffle(inDF)
maxVals = inDF.max()
minVals = inDF.min()
#inDF=(inDF-inDF.min())/(inDF.max()-inDF.min())
inDF = (inDF-minVals)/(maxVals-minVals)

normFactors = [maxVals.drop(['match']), minVals.drop(['match'])]
#normFactors = np.asarray(maxVals.drop(['match']))
np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

train, test = train_test_split(inDF, test_size=0.1)

y_train = train['match']
y_test = test['match']

train = train.drop(['match'],axis=1)
test = test.drop(['match'],axis=1)

test, train = test.values, train.values
'''
Things to tune:
layers, nodes
activation function (leakyReLu is probably best)
optimizer (Nadam?)
learning rate, beta_1, beta_2
dropout (none seems best)
weight initialization
'''
#def create_model(layers=[125,125,75,75,50,50,50,50], activation='LeakyReLU', regularizer=None):
def create_model(layers, nodes, activation='LeakyReLU', regularizer=None):
    from tensorflow.keras.models import Sequential # feed-forward neural network (sequential layers)
    from tensorflow.keras.layers import Dense, Dropout, LeakyReLU # fully interconnected layers
    model = Sequential()
    #model.add(Dense(layers[0], input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(Dense(nodes, input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    # hidden layer: 5 nodes by default
    #for l in range(layers):
    for l in range(layers):
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        model.add(Dense(nodes, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
        #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
param_grid = {'layers':[4,5,6, 7],
              #'layers':[(75,75,75,75,75,75),
              #          (125,125,75,75,50,50),
              #          (100,100,100,100,100,100), 
              #          (150,150,75,75,25,25)]
              'nodes': [40, 50, 60, 70],
              #'batch_size': [32, 64, 128],
              'epochs':[40]
              #'regularizer':
}
model=KerasClassifier(build_fn=create_model, verbose=1)
#model=KerasRegressor(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=5)
grid_result = grid.fit(train, y_train)

resultFile.write('Best params: ')
resultFile.write(json.dumps(grid_result.best_params_))
resultFile.write('\n')

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
    resultFile.write("%0.3f (+/-%0.03f) for %r \n"
          % (mean, std * 2, params))

resultFile.close()

