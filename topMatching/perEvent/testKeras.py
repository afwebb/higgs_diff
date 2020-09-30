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
from tensorflow.keras.models import load_model
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
#outDir = sys.argv[2]

topModel = load_model(sys.argv[2])

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

topModel = load_model(sys.argv[2])
topModel.compile(loss="categorical_crossentropy", optimizer='adam')

y_train_pred = topModel.predict(train)
y_test_pred = topModel.predict(test)

y_train_pred_best = [sorted(x.argsort()[-2:]) for x in y_train_pred]
y_train_best = [sorted(x.argsort()[-2:]) for x in y_train]

print(y_train_pred.shape, y_test_pred.shape)
print(y_train.shape, y_test.shape)
print(y_train_pred_best[:10], y_train_best[:10])

nCorrect, nCorrect1 = 0, 0
for yT, yP in zip(y_train_best, y_train_pred_best):
    if yP[0] in yT and yP[1] in yT:
        nCorrect+=1
    if yP[0] in yT or yP[1] in yT:
        nCorrect1+=1
        
print('Correct', nCorrect/len(y_train_best), nCorrect1/len(y_train_best) )

confMat = sklearn.metrics.confusion_matrix(y_train, y_train_pred)                                    
plt.figure()                                                                                                                
ax = plt.subplot()                                                                                                     
sns.heatmap(confMat, annot=True, robust=True)
ax.set_xlabel("Predicted")
ax.set_ylabel("Truth")
ax.set_title(f"{alg.capitalize()} Confusion Matrix")
plt.savefig(f'plots/{outDir}/conf_matrix.png')

#make_plots('keras', result, outDir, y_train, y_test, y_train_pred, y_test_pred)
