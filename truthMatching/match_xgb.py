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


inFile = sys.argv[1]
inDF = pd.read_csv(inFile)

outStr = sys.argv[2]

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
    'max_depth': 21,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.7,
    'colsample_bytree' : 0.7,
    'eval_metric': 'auc',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=1000, verbose_eval=True)

best_nrounds = pd.Series.idxmax(gbm['test-auc-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open('models/xgb_match_'+outStr+'.dat', "wb"), protocol=2)
#bst.save_model('models/xgb_match_test.model')

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)
train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)

testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

plt.figure()
plt.hist(testPredTrue, 30, log=False, alpha=0.5, label='Correct')
plt.hist(testPredFalse[:len(testPredTrue)], 30, log=False, alpha=0.5, label='Incorrect')
plt.title("BDT Output, Test Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/match_xgb_'+outStr+'_test_score.png')

plt.figure()
plt.hist(trainPredTrue, 30, log=False, alpha=0.5, label='Correct')
plt.hist(trainPredFalse[:len(trainPredTrue)], 30, log=False, alpha=0.5, label='Incorrect')
plt.title("BDT Output, Train Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/match_xgb_'+outStr+'_train_score.png')


plt.figure()
fip = xgb.plot_importance(bst)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/match_xgb_'+outStr+'_feature_importance.png')

auc = sk.metrics.roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred)

plt.figure()
plt.plot(fpr, tpr, label='AUC = %.3f' %(auc))
plt.title('xgb match, AUC = %.3f' %(auc))

plt.savefig('plots/match_xgb_'+outStr+'_roc.png')

y_test_bin = np.where(y_test_pred > 0.5, 1, 0)
print(y_test_bin)
print('Confusion Matrix:', sklearn.metrics.confusion_matrix(y_test, y_test_bin))

