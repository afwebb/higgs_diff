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
import pickle
import sys
import torch
import scipy
from matchPlots import make_plots

inFile = sys.argv[1]
inDF = pd.read_csv(inFile, nrows=1e6)

maxDepth = 10

outDir = sys.argv[2]

inDF = sk.utils.shuffle(inDF)
inDF[abs(inDF) < 0.01] = 0
train, test = train_test_split(inDF, test_size=0.3)

y_train = train['match']
y_test = test['match']

train = train.drop(['match'],axis=1)
test = test.drop(['match'],axis=1)

xgb_train = xgb.DMatrix(train, label=y_train, feature_names=list(train))
xgb_test = xgb.DMatrix(test, label=y_test, feature_names=list(train))

params = {
    'learning_rate' : 0.01,
    'max_depth': maxDepth,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.6,
    'colsample_bytree' : 0.6,
    'eval_metric': 'auc',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=250, verbose_eval=True)

best_nrounds = pd.Series.idxmax(gbm['test-auc-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open('models/xgb_match_'+outDir+'.dat', "wb"), protocol=2)
#bst.save_model('models/xgb_match_test.model')

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)
train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)

testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

#plt.figure()                                                                                           
#fip = xgb.plot_importance(bst)                                                                                              
#plt.title("xgboost feature important")                                                                                     
#plt.legend(loc='lower right')                                                                                             
#plt.savefig('plots/'+outDir+'/xgb_feature_importance.png')

make_plots('xgb', bst, outDir, y_train, y_test, y_train_pred, y_test_pred)
