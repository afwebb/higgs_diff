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

epochs = 50
nodes = 75
layers = 8
'''
if outDir=='higgs2lSS' or outDir=='higgsTop2lSS':
    epochs = 100
    layers = 10
    nodes = 50
elif outDir=='higgs3lS' or outDir=='higgsTop3lS':
    epochs = 100
    layers = 8
    nodes = 60
elif outDir=='higgs3lF' or outDir=='higgsTop3lF':
    epochs = 100
    layers = 10
    nodes = 60
'''
inDF = pd.read_csv(inFile, index_col=False)
inDF = sk.utils.shuffle(inDF)
maxVals = inDF.max()
minVals = inDF.min()
#inDF=(inDF-inDF.min())/(inDF.max()-inDF.min())                                                                                  
yMax = inDF['higgs_pt'].max() 

inDF = (inDF-minVals)/(maxVals-minVals)

normFactors = [maxVals.drop(['higgs_pt']), minVals.drop(['higgs_pt']), yMax]
#normFactors = np.asarray(maxVals.drop(['match']))                                                                                
np.save('models/'+outDir+'_normFactors.npy', normFactors)

nFeatures = len(list(inDF))-1

train, test = train_test_split(inDF, test_size=0.2)

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

test, train = test.values, train.values

#layers = (125,125,75,75,50,50,50,25,25)
#layers = (125,125,125,50,50,50,50,50)
def create_model(layers=layers, nodes=nodes, regularizer=None, activation='relu'):
    from keras.models import Sequential # feed-forward neural network (sequential layers)
    from keras.layers import Dense, Dropout, LeakyReLU  # fully interconnected layers
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
    # one output, mapped to [0,1] by sigmoid function
    model.add(Dense(1, activation='sigmoid'))
    # assemble the model (Translate to TensorFlow)
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model=KerasRegressor(build_fn=create_model, verbose=1)
result=model.fit(train, y_train, epochs=epochs)

model.model.save("models/keras_model_"+outDir+".h5")

y_pred_train = model.predict(train)
y_pred_test = model.predict(test)

y_train = y_train*yMax
y_test = y_test*yMax
y_train_pred = y_pred_train*yMax
y_test_pred = y_pred_test*yMax

makePlots('keras', model, outDir, y_train, y_test, y_train_pred, y_test_pred)

'''
test_loss = np.sqrt(sk.metrics.mean_squared_error(y_test, y_test_pred))
train_loss = np.sqrt(sk.metrics.mean_squared_error(y_train, y_train_pred))

#ROC curve                                                                                                                       
c = 150000

plt.figure()

yTrain = np.where(y_train > c, 1, 0)
ypTrain = y_train_pred
auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)
plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))

yTest = np.where(y_test > c, 1, 0)
ypTest = y_test_pred
auc = sk.metrics.roc_auc_score(yTest,ypTest)
fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)
plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

plt.title("Keras ROC")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/keras_roc.png')

# Calculate the point density                                                                                          
scatterSize = 60000
xy = np.vstack([y_test[:scatterSize], y_test_pred[:scatterSize]])
z = scipy.stats.gaussian_kde(xy)(xy)
#z = scipy.stats.gaussian_kde(np.vstack([y_test, y_test_pred]))(np.vstack([y_test, y_test_pred]))                                 

plt.figure()
plt.scatter(y_test[:scatterSize]/1000, y_test_pred[:scatterSize]/1000, c=np.log(z), edgecolor='')
plt.title("Keras Test Data, MSE=%f" %(test_loss))
plt.xlabel('Truth $p_T$ [GeV]')
plt.ylabel('Predicted $p_T$ [GeV]')
plt.xlim(1,1000)
plt.ylim(1,1000)
plt.plot([0,1000],[0,1000],zorder=10)
plt.savefig('plots/'+outDir+'/keras_test_pt_scatter.png')

# Calculate the point density                                                                                                     
xy = np.vstack([y_train[:scatterSize], y_train_pred[:scatterSize]])
z = scipy.stats.gaussian_kde(xy)(xy)
#z = scipy.stats.gaussian_kde(np.vstack([y_train, y_train_pred]))(np.vstack([y_train, y_train_pred]))                             

plt.figure()
plt.scatter(y_train[:scatterSize]/1000, y_train_pred[:scatterSize]/1000, c=np.log(z), edgecolor='')
plt.title("Keras Train Data, MSE=%f" %(train_loss))
plt.xlabel('Truth $p_T$ [GeV]')
plt.ylabel('Predicted $p_T$ [GeV]')
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.plot([0,1000],[0,1000],zorder=10)
plt.savefig('plots/'+outDir+'/keras_train_pt_scatter.png')

errBins = np.linspace(0, 1200000, num=30)
testErrVec = np.array(np.sqrt((y_test_pred - y_test)**2))
print(testErrVec)
y_pred_bins = []
y_err_bins = []

for b in range(1, len(errBins)):
    binned_y = testErrVec[(y_test<errBins[b]) & (y_test>errBins[b-1])]
    y_pred_bins.append( np.mean( binned_y ) )
    y_err_bins.append(  np.std(binned_y) )

binned_y = testErrVec[(y_test>1200000)]
y_pred_bins.append( np.mean( binned_y ) )
y_err_bins.append(  np.std(binned_y) )
errBins = np.append( errBins, np.mean(y_test[y_test>1200000]) )

print(len(errBins), len(y_pred_bins), len(y_err_bins))

errBins = np.divide(errBins, 1000)
y_pred_bins = np.divide(y_pred_bins, 1000)
y_err_bins = np.divide(y_err_bins, 1000)

plt.figure()
plt.errorbar( errBins[1:], y_pred_bins, yerr=y_err_bins)
plt.plot([0,1200],[0,1200])#, zorder=10)
plt.title('Prediction Error')
plt.xlabel('Truth Higgs Pt [MeV]')
plt.ylabel('RMSE')
plt.savefig('plots/'+outDir+'/keras_err.png')
'''
