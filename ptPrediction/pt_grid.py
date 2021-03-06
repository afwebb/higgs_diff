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
from keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy
import json

inFile = sys.argv[1]
outDir = sys.argv[2]

resultFile = open('models/gridResults_'+outDir+'.txt', 'w')

inDF = pd.read_csv(inFile, index_col=False)
inDF = sk.utils.shuffle(inDF)
meanVals = inDF.mean()
maxVals = inDF.max()
minVals = inDF.min()
#inDF=(inDF-inDF.min())/(inDF.max()-inDF.min())
inDF = (inDF-meanVals)/(maxVals-minVals)

normFactors = [maxVals.drop(['higgs_pt']), minVals.drop(['higgs_pt'])]
#normFactors = np.asarray(maxVals.drop(['higgs_pt']))
#np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

#train, test = train_test_split(inDF, test_size=0.1)
train = inDF

y_train = train['higgs_pt']
#y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
#test = test.drop(['higgs_pt'],axis=1)

#test, train = test.values, train.values
train = train.values
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
def create_model(layers, nodes, learning_rate, activation='LeakyReLU', regularizer=None):
    from keras.models import Sequential # feed-forward neural network (sequential layers)
    from keras.layers import Dense, Dropout, LeakyReLU # fully interconnected layers
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
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
param_grid = {'layers':[5, 6, 7, 8,10],
              #'layers':[(75,75,75,75,75,75),
              #          (125,125,75,75,50,50),
              #          (100,100,100,100,100,100), 
              #          (150,150,75,75,25,25)]
              'nodes': [40, 50, 70, 100],
              #'batch_size': [32, 64, 128],
              'epochs':[120],#[20,40,60,80]
              'learning_rate':[0.001],
              #'regularizer':
}
#model=KerasClassifier(build_fn=create_model, verbose=1)
model=KerasRegressor(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=12)
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

