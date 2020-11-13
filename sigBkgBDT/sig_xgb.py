#Script for seperating high pt and low pt events using XGBoost
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import sys
import torch
import scipy
from matchPlots import make_plots
from dfCuts import dfCuts

inFile = sys.argv[1]
outDir = sys.argv[2]
#dsid = sys.argv[1]
#njet = sys.argv[2]

#inDF = pd.read_csv('../inputData/345673_4j.csv')

inDF = pd.read_csv(inFile)
inDF = inDF.dropna()

inDF = dfCuts(inDF, outDir)

inDF = sk.utils.shuffle(inDF)

weights = inDF[['weight']].values.astype(float)
weights = preprocessing.MinMaxScaler().fit_transform(weights)
inDF['weight'] = weights

print(inDF.shape)

train, test = train_test_split(inDF, test_size=0.2)

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

xgb_train = xgb.DMatrix(train, label=y_train, feature_names=list(test), weight=weights_train)
xgb_test = xgb.DMatrix(test, label=y_test, feature_names=list(train), weight=weights_test)

params = {
    'learning_rate' : 0.005,
    'max_depth': 6,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'eval_metric': 'auc',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=1500, verbose_eval=True)

best_nrounds = pd.Series.idxmax(gbm['test-auc-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open("models/"+outDir.replace('/','_')+".dat", "wb"))

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

test_loss = sk.metrics.roc_auc_score(y_test, y_test_pred)
train_loss = sk.metrics.roc_auc_score(y_train, y_train_pred)

make_plots('xgb', bst, outDir, y_train, y_test, y_train_pred, y_test_pred, weights_train, weights_test)
