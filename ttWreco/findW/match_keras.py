import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy

inFile = sys.argv[1]
outDir = sys.argv[2]

#Use optimal parameters obtained from grid search
#if outDir=='higgs':
epochs = 75
layers = 6
nodes = 75

inDF = pd.read_csv(inFile, index_col=False)
#inDF = inDF.drop(['lep_Parent_0'],axis=1)
#inDF = inDF.drop(['lep_Parent_1'], axis=1)
inDF = sk.utils.shuffle(inDF)
maxVals = inDF.max()
minVals = inDF.min()
#inDF=(inDF-inDF.min())/(inDF.max()-inDF.min())
inDF = (inDF-minVals)/(maxVals-minVals)

normFactors = [maxVals.drop(['fromW']), minVals.drop(['fromW'])]
#normFactors = np.asarray(maxVals.drop(['fromW']))
np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

train, test = train_test_split(inDF, test_size=0.2)

y_train = train['fromW']
y_test = test['fromW']

train = train.drop(['fromW'],axis=1)
test = test.drop(['fromW'],axis=1)

test, train = test.values, train.values

def create_model(layers=layers, nodes=nodes, activation='LeakyReLU', regularizer=None):
    from keras.models import Sequential # feed-forward neural network (sequential layers)
    from keras.layers import Dense, Dropout, LeakyReLU # fully interconnected layers
    model = Sequential()
    model.add(Dense(nodes, input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    # hidden layer: 5 nodes by default
    for l in range(layers):
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        model.add(Dense(nodes, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
        #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#model=KerasClassifier(build_fn=create_model, verbose=1)
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, validation_split=0.2, epochs=epochs)

model.model.save("models/keras_model_"+outDir+".h5")

y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

print(y_test_pred)

test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)
train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)

testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

minLen = min([len(testPredTrue), len(testPredFalse), len(trainPredTrue), len(trainPredFalse)])

plt.figure()
plt.hist(testPredTrue[:minLen], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='from W - Test')
plt.hist(testPredFalse[:minLen], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='from Top - Test')
plt.hist(trainPredTrue[:minLen], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='from W - Train')
plt.hist(trainPredFalse[:minLen], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='from Top - Train')
plt.title("W Lepton Matching Output")
plt.xlabel('Keras Score')
plt.ylabel('NEvents')
plt.legend(loc='upper center')
plt.savefig('plots/'+outDir+'/keras_score.png')

plt.figure()
auc = sk.metrics.roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred)
plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

auc = sk.metrics.roc_auc_score(y_train, y_train_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_train, y_train_pred)
plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))
plt.legend(loc='lower right')
plt.title('Keras Match ROC')
plt.savefig('plots/'+outDir+'/keras_roc.png')

y_test_bin = np.where(y_test_pred > 0.5, 1, 0)
print(y_test_bin)
print('Confusion Matrix:', sklearn.metrics.confusion_matrix(y_test, y_test_bin))
