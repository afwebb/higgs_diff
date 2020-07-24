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

#outStr = sys.argv[1]
max_depth=4

inFile = sys.argv[1]
inDF = pd.read_csv(inFile)

outDir = sys.argv[2]

#if outDir=='fullLep':
inDF['decay'] = inDF['decay'].replace({0:1, 1:0})

inDF = sk.utils.shuffle(inDF)
inDF[abs(inDF) < 0.01] = 0
train, test = train_test_split(inDF, test_size=0.3)

y_train = train['decay']
y_test = test['decay']

train = train.drop(['decay'],axis=1)
test = test.drop(['decay'],axis=1)

xgb_train = xgb.DMatrix(train, label=y_train, feature_names=list(train))
xgb_test = xgb.DMatrix(test, label=y_test, feature_names=list(train))
'''
#features = ["Mll01","Mll02","Mll12","Mlll","Mlt00","Mlt01","Mlt10","Mlt11","Mlt20","Mlt21","dRll01","dRll02","dRll12","dRlt00","dRlt01","dRlt10","dRlt11","dRlt20","dRlt21","lep_Pt_0","lep_Pt_1","lep_Pt_2","met","met_phi","nJets","nJets_MV2c10_70","top_Pt_0","top_Pt_1"]

features = ['Mll01', 'Mll02', 'Mll12', 'Mlll', 'Mlt00', 'Mlt01', 'Mlt10', 'Mlt11', 'Mlt20', 'Mlt21', 'dRll01', 'dRll02', 'dRll12', 'dRlt00', 'dRlt01', 'dRlt10', 'dRlt11', 'dRlt20', 'dRlt21', 'lep_Pt_0', 'lep_Pt_1', 'lep_Pt_2', 'met', 'met_phi', 'nJets', 'nJets_MV2c10_70', 'top_Pt_0', 'top_Pt_1']

xgb_test = xgb.DMatrix('tensors/xgb_test_'+outStr+'.buffer',
                       feature_names = features)
xgb_train = xgb.DMatrix('tensors/xgb_train_'+outStr+'.buffer',
                        feature_names = features)

y_test = torch.load('tensors/torch_y_test_'+outStr+'.pt')
y_test = y_test.float().detach().numpy()
y_train = torch.load('tensors/torch_y_train_'+outStr+'.pt')
y_train = y_train.float().detach().numpy()
'''
params = {
    'learning_rate' : 0.01,
    'max_depth': max_depth,
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
pickle.dump(bst, open('models/xgb_decay_'+outDir+'.dat', "wb"), protocol=2)
#bst.save_model('models/xgb_decay_test.model')

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)
train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)

testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

plt.figure()
plt.hist(testPredTrue, 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Semi-lep - Test')
plt.hist(testPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Full-lep - Test')
plt.hist(trainPredTrue[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Semi-lep - Train')
plt.hist(trainPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Full-lep - Train')
plt.title("BDT Output, max depth=%i" %(max_depth))
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/xgb_score.png')

plt.figure()
fip = xgb.plot_importance(bst)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/xgb_feature_importance.png')

plt.figure()
auc = sk.metrics.roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred)
plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

auc = sk.metrics.roc_auc_score(y_train, y_train_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_train, y_train_pred)
plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))
plt.legend(loc='lower right')
plt.title('XGBoost Decay ROC')
plt.savefig('plots/'+outDir+'/xgb_roc.png')

y_test_bin = np.where(y_test_pred > 0.5, 1, 0)
print(y_test_bin)
print('Confusion Matrix:', sklearn.metrics.confusion_matrix(y_test, y_test_bin))

'''
plt.figure()
plt.hist(testPredTrue, 30, log=False, alpha=0.5, label='Semi-leptonic')
plt.hist(testPredFalse[:len(testPredTrue)], 30, log=False, alpha=0.5, label='Fully leptonic')
plt.title("BDT Output, Test Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/decay_xgb_'+outStr+'_test_score.png')

plt.figure()
plt.hist(trainPredTrue, 30, log=False, alpha=0.5, label='Semi-leptons')
plt.hist(trainPredFalse[:len(trainPredTrue)], 30, log=False, alpha=0.5, label='Fully leptonic')
plt.title("BDT Output, Train Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/decay_xgb_'+outStr+'_train_score.png')


plt.figure()
fip = xgb.plot_importance(bst)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/decay_xgb_'+outStr+'_feature_importance.png')

auc = sk.metrics.roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred)

plt.figure()
plt.plot(fpr, tpr, label='AUC = %.3f' %(auc))
plt.title('xgb decay, AUC = %.3f' %(auc))

plt.savefig('plots/decay_xgb_'+outStr+'_roc.png')

y_test_bin = np.where(y_test_pred > 0.5, 1, 0)
print(y_test_bin)
print('Confusion Matrix:', sklearn.metrics.confusion_matrix(y_test, y_test_bin))

'''
