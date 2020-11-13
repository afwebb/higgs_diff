import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential # feed-forward neural network (sequential layers)
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization # fully interconnected layers              
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy
from matchPlots import make_plots
from dfCuts import dfCuts

inFile = sys.argv[1]
outDir = sys.argv[2]

inDF = pd.read_csv(inFile, index_col=False)
inDF = inDF.dropna()
inDF = sk.utils.shuffle(inDF)

inDF = dfCuts(inDF, outDir)

maxVals = inDF.max()
minVals = inDF.min()

inDF = (inDF-minVals)/(maxVals-minVals)
weights = inDF[['weight']].values.astype(float)
weights = preprocessing.MinMaxScaler().fit_transform(weights)

normFactors = [maxVals.drop(['signal']), minVals.drop(['signal'])]
np.save('models/'+outDir+'_normFactors.npy', normFactors)

train, test = train_test_split(inDF, test_size=0.1)

y_train = train['signal']
y_test = test['signal']

train = train.drop(['signal'],axis=1)
test = test.drop(['signal'],axis=1)

weights_train = train[['weight']].values.astype(float)
weights_train = preprocessing.MinMaxScaler().fit_transform(weights_train).flatten()
weights_test = test[['weight']].values.astype(float)
weights_test = preprocessing.MinMaxScaler().fit_transform(weights_test).flatten()

train = train.drop(['weight'], axis=1)
test = test.drop(['weight'], axis=1)
nFeatures = len(list(train))

test, train = test.values, train.values

def create_model(layers=(75,75,75,75,75,75), activation='LeakyReLU', regularizer=None):
    model = Sequential()
    model.add(Dense(layers[0], input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    # hidden layer: 5 nodes by default
    for l in layers:
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        model.add(Dense(l, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
        model.add(BatchNormalization())
        #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['AUC'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#model=KerasClassifier(build_fn=create_model, verbose=1)
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, sample_weight=weights_train, epochs=120)

model.model.save("models/keras_model_"+outDir+".h5")

y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

make_plots('keras', result, outDir, y_train, y_test, y_train_pred, y_test_pred, weights_train, weights_test)
