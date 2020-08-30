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

if outDir=='higgs' or outDir=='higgsBin':
    epochs = 30
    layers = 10
    nodes = 50
elif outDir=='higgs1l' or outDir=='higgs1lBin':
    epochs = 30
    layers = 8
    nodes = 50
elif outDir=='higgs2l' or outDir=='higgs2lBin':
    epochs = 50
    layers = 10
    nodes = 50
else:
    epochs = 60
    layers = 6
    nodes = 75

inDF = pd.read_csv(inFile, index_col=False)
#inDF['higgs_pt'] = pd.cut(inDF['higgs_pt'], bins=[0, 150000, 9999999999], labels=[0,1])
inDF['higgs_pt'][inDF.higgs_pt<150000] = 0
inDF['higgs_pt'][inDF.higgs_pt>150000] = 1

print(inDF['higgs_pt'])
inDF = sk.utils.shuffle(inDF)
maxVals = inDF.max()
minVals = inDF.min()
inDF = (inDF-minVals)/(maxVals-minVals)

normFactors = [maxVals.drop(['higgs_pt']), minVals.drop(['higgs_pt'])]
np.save('models/bin_'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

train, test = train_test_split(inDF, test_size=0.2)

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

test, train = test.values, train.values

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
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['AUC'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, epochs=epochs)

model.model.save("models/bin_keras_model_"+outDir+".h5")

y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

makePlots(bin_keras, model, outDir, y_train, y_test, y_train_pred, y_test_pred)

'''
test_loss = np.sqrt(sk.metrics.mean_squared_error(y_test, y_test_pred))
train_loss = np.sqrt(sk.metrics.mean_squared_error(y_train, y_train_pred))

#ROC curve                                                                                                                       
#c = 150000

plt.figure()

yTrain = y_train
ypTrain = y_train_pred
auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)
plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))

yTest = y_test
ypTest = y_test_pred
auc = sk.metrics.roc_auc_score(yTest,ypTest)
fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)
plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

plt.title("Keras ROC")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/bin_keras_roc.png')

#Scores 
testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

plt.figure()
plt.hist(testPredTrue, 30, range=(-0.1,1.1), log=False, alpha=0.5, label='High Pt - Test')
plt.hist(testPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Low Pt - Test')
plt.hist(trainPredTrue[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='High Pt - Train')
plt.hist(trainPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Low Pt - Train')
plt.title("Keras Output")
plt.xlabel('Keras Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/bin_keras_score.png')
'''
