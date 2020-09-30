import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # feed-forward neural network (sequential layers)                         
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU # fully interconnected layers  
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
from ast import literal_eval

tf.keras.backend.set_floatx('float32')

inFile = sys.argv[1]
outDir = sys.argv[2]

#Use optimal parameters obtained from grid search
best_params = {"epochs": 100, "layers": 6, "nodes": 120}

print(best_params)

inDF = pd.read_csv(inFile, index_col=False)
inDF = sk.utils.shuffle(inDF)
inDF.loc[:,'match'] = inDF.loc[:,'match'].apply(lambda x: literal_eval(x))

maxVals = inDF.max()
minVals = inDF.min()

maxVals = maxVals.drop(['match'])
minVals = minVals.drop(['match'])
#min_max_scaler = sk.preprocessing.MinMaxScaler()

#inDF=(inDF-inDF.min())/(inDF.max()-inDF.min())
#inDF = pd.DataFrame(min_max_scaler.fit_transform(inDF.values))#(inDF-minVals)/(maxVals-minVals)

normFactors = [maxVals, minVals]
#normFactors = np.asarray(maxVals.drop(['match']))
np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1
print(list(inDF))
train, test = train_test_split(inDF, test_size=0.1)
 
y_train = np.vstack(train['match'])
y_test = np.vstack(test['match'])
print(y_train.shape)
outDim = y_train.shape[1]

train = train.drop(['match'],axis=1)
test = test.drop(['match'],axis=1)

train=(train-minVals)/(maxVals-minVals)
test=(test-minVals)/(maxVals-minVals) 

test, train = test.values, train.values

layers=best_params['layers']
nodes=best_params['nodes']
activation='LeakyReLU'
regularizer=None
#def create_model(layers=best_params['layers'], nodes=best_params['nodes'], activation='LeakyReLU', regularizer=None):
 
model = Sequential()
model.add(Dense(nodes, input_dim = nFeatures, kernel_regularizer=regularizer))
model.add(LeakyReLU(alpha=0.05))
# hidden layer: 5 nodes by default
for l in range(layers):
    #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(nodes, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    #model.add(Dropout(0.2))
model.add(Dense(outDim, activation='softmax'))
#model.add(Dense(1, activation=activation))
model.compile(loss="categorical_crossentropy", optimizer='adam')
#model.compile(loss="mean_squared_error", optimizer='adam') 
print(model.summary())
#return model

#model=KerasClassifier(build_fn=create_model, verbose=1)
#model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, validation_split=0.1, epochs=best_params['epochs'])

model.save("models/keras_model_"+outDir+".h5")

y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

y_train_pred_best = [sorted(x.argsort()[-2:]) for x in y_train_pred]
y_train_best = [sorted(x.argsort()[-2:]) for x in y_train]

print(y_train_pred.shape, y_test_pred.shape)
print(y_train.shape, y_test.shape)
print(y_train_pred_best[:10], y_train_best[:10])

nCorrect1 = [x[0]==y[0] for x, y in zip(y_train_best, y_train_pred_best)]
nCorrect2 = [x[0]==y[0] for x, y in zip(y_train_best, y_train_pred_best)]
print('Correct', sum(nCorrect)/len(y_train) )

confMat = sklearn.metrics.confusion_matrix(y_train, y_train_pred)                                    
plt.figure()                                                                                                                
ax = plt.subplot()                                                                                                     
sns.heatmap(confMat, annot=True, robust=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("Truth")
ax.set_title(f"{alg.capitalize()} Confusion Matrix")
plt.savefig(f'plots/{outDir}/conf_matrix.png')

#make_plots('keras', result, outDir, y_train, y_test, y_train_pred, y_test_pred)
