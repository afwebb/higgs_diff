import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy

inFile = sys.argv[1]
outDir = sys.argv[2]

inDF = pd.read_csv(inFile, index_col=False)
#inDF = inDF.dropna()
inDF = sk.utils.shuffle(inDF)

if outDir=="2lHigh":
    inDF = inDF[inDF['pt_score']>150000]
    #inDF = inDF.drop(['pt_score'],axis=1)
    #inDF = inDF.drop(['bin_score'],axis=1)
elif outDir=='2lLow':
    inDF = inDF[inDF['pt_score']<150000]
    #inDF = inDF.drop(['pt_score'],axis=1)
    #inDF = inDF.drop(['bin_score'],axis=1)

if outDir=="3lF":
    inDF = inDF[inDF['decayScore']>0.18]
    inDF = inDF.drop(['pt_score_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)
elif outDir=="3lS":
    inDF = inDF[inDF['decayScore']<0.18]
    inDF = inDF.drop(['pt_score_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)

if outDir=="3lFHigh":
    inDF = inDF[inDF['decayScore']>0.18]
    inDF = inDF[inDF['pt_score_3lF']>150000]
    #inDF = inDF.drop(['pt_score_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)                                                                                   inDF = inDF.drop(['pt_score_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)
elif outDir=="3lSHigh":
    inDF = inDF[inDF['decayScore']<0.18]
    inDF = inDF[inDF['pt_score_3lS']>150000]
    inDF = inDF.drop(['pt_score_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)
    #inDF = inDF.drop(['pt_score_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)

if outDir=="3lFLow":
    inDF = inDF[inDF['decayScore']>0.18]
    inDF = inDF[inDF['pt_score_3lF']<150000]
    #inDF = inDF.drop(['pt_score_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)
    inDF = inDF.drop(['pt_score_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)
elif outDir=="3lSLow":
    inDF = inDF[inDF['decayScore']<0.18]
    inDF = inDF[inDF['pt_score_3lS']<150000]
    inDF = inDF.drop(['pt_score_3lF'],axis=1)      

maxVals = inDF.max()
minVals = inDF.min()
inDF = (inDF-minVals)/(maxVals-minVals)
#weights = inDF[['scale_nom']].values.astype(float)
#weights = preprocessing.MinMaxScaler().fit_transform(weights)
#inDF['scale_nom'] = weights

normFactors = [maxVals.drop(['signal']), minVals.drop(['signal'])]
#normFactors = np.asarray(maxVals.drop(['match']))
np.save('models/'+outDir+'_normFactors.npy', normFactors)

train, test = train_test_split(inDF, test_size=0.2)

y_train = train['signal']
y_test = test['signal']

train = train.drop(['signal'],axis=1)
test = test.drop(['signal'],axis=1)

weight_train = train['scale_nom']
weight_test = test['scale_nom']
train = train.drop(['scale_nom'], axis=1)
test = test.drop(['scale_nom'], axis=1)
nFeatures = len(list(train))

test, train = test.values, train.values

def create_model(layers=(100,100,75,75,50,50), activation='LeakyReLU', regularizer=None):
    from keras.models import Sequential # feed-forward neural network (sequential layers)
    from keras.layers import Dense, Dropout, LeakyReLU # fully interconnected layers
    model = Sequential()
    model.add(Dense(layers[0], input_dim = nFeatures, kernel_regularizer=regularizer))
    model.add(LeakyReLU(alpha=0.05))
    # hidden layer: 5 nodes by default
    for l in layers:
        #model.add(Dense(l, activation=activation, kernel_regularizer=regularizer))
        model.add(Dense(l, kernel_regularizer=regularizer))
        model.add(LeakyReLU(alpha=0.05))
        #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#model=KerasClassifier(build_fn=create_model, verbose=1)
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, validation_split=0.2, epochs=50)

model.model.save("models/keras_var_model_"+outDir+".h5")

y_train_pred = model.predict(train)
y_test_pred = model.predict(test)

print(y_test_pred)

test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)
train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)

testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

plt.figure()
plt.hist(testPredTrue, 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Correct - Test')
plt.hist(testPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Incorrect - Test')
plt.hist(trainPredTrue[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Correct - Train')
plt.hist(trainPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Incorrect - Train')
plt.title("Keras Output")
plt.xlabel('Keras Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/keras_var_score.png')

plt.figure()
auc = sk.metrics.roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred)
plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

auc = sk.metrics.roc_auc_score(y_train, y_train_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_train, y_train_pred)
plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))
plt.legend(loc='lower right')
plt.title('Keras Match ROC')
plt.savefig('plots/'+outDir+'/keras_var_roc.png')

y_test_bin = np.where(y_test_pred > 0.5, 1, 0)
print(y_test_bin)
print('Confusion Matrix:', sklearn.metrics.confusion_matrix(y_test, y_test_bin))
