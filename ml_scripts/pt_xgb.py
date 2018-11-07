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

dsid = sys.argv[1]
njet = sys.argv[2]

#inDF = pd.read_csv('../inputData/345673_4j.csv')

inDF = pd.read_csv('../inputData/'+dsid+'_'+njet+'.csv')

cutoff = 300000

dsid = sys.argv[1]
njet = sys.argv[2]

#inDF = inDF[inDF['is2LSS0Tau']==0]

#inDF.loc[ inDF['higgs_pt'] <= cutoff, 'higgs_pt'] = 0 ;
#inDF.loc[ inDF['higgs_pt'] > cutoff, 'higgs_pt'] = 1 ;

train, test = train_test_split(inDF, test_size=0.3)

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

xgb_train = xgb.DMatrix(train, label=y_train, feature_names=list(test))
xgb_test = xgb.DMatrix(test, label=y_test, feature_names=list(train))
'''

xgb_test = xgb.DMatrix('../inputData/tensors/xgb_test_'+dsid+'_'+njet+'.buffer')
xgb_train = xgb.DMatrix('../inputData/tensors/xgb_train_'+dsid+'_'+njet+'.buffer')

y_test = torch.load('../inputData/tensors/torch_y_test_'+dsid+'_'+njet+'.pt')
y_test = y_test.float().detach().numpy()
y_train = torch.load('../inputData/tensors/torch_y_train_'+dsid+'_'+njet+'.pt')
y_train = y_train.float().detach().numpy()
'''
params = {
    'learning_rate' : 0.1,
    'max_depth': 12,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'eval_metric': 'mae',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=200, verbose_eval=True)

best_nrounds = pd.Series.idxmin(gbm['test-mae-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open("xgb_models/"+njet+'_'+dsid+".dat", "wb"))

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

cutoff = [150000, 200000, 250000]

plt.figure()
for c in cutoff:
    yTrain = np.where(y_train > c, 1, 0)
    ypTrain = y_train_pred

    auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
    fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)

    plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

plt.title("xgb Train ROC")
plt.legend(loc='lower right')    
plt.savefig('plots/'+njet+'_'+dsid+'/xgb_train_roc.png')

plt.figure()
for c in cutoff:
    yTest = np.where(y_test > c, 1, 0)
    ypTest = y_test_pred

    auc = sk.metrics.roc_auc_score(yTest,ypTest)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

    plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

plt.title("xgb Test ROC")
plt.legend(loc='lower right')    
plt.savefig('plots/'+njet+'_'+dsid+'/xgb_test_roc.png')

plt.figure()
fip = xgb.plot_importance(bst)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/'+njet+'_'+dsid+'/xgb_feature_importance.png')

plt.figure()
plt.hist(y_train, 20, log=False, range=(0,800000), alpha=0.5, label='truth')
plt.hist(y_train_pred,20, log=False, range=(0,800000), alpha=0.5, label='train')
plt.legend(loc='upper right')
plt.title("XGBoost Train Data")
plt.xlabel('Higgs Pt')
plt.ylabel('NEvents')
plt.savefig('plots/'+njet+'_'+dsid+'/xgb_train_pt_spectrum.png')

plt.figure()
plt.hist(y_test, 20, log=False, range=(0,800000), alpha=0.5, label='truth')
plt.hist(y_test_pred,20, log=False, range=(0,800000), alpha=0.5, label='test')
plt.legend(loc='upper right')
plt.title("XGBoost Test Data")
plt.xlabel('Higgs Pt')
plt.ylabel('NEvents')
plt.savefig('plots/'+njet+'_'+dsid+'/xgb_test_pt_spectrum.png')


# Calculate the point density                                                                                 xy = np.vstack([y_test, y_test_pred])
#z = scipy.stats.gaussian_kde(xy)(xy)
z = scipy.stats.gaussian_kde(np.vstack([y_test, y_test_pred]))(np.vstack([y_test, y_test_pred]))

plt.figure()
plt.scatter(y_test, y_test_pred, c=z, edgecolor='')
plt.title("XGBoost Test Data")
plt.xlabel('Truth Pt')
plt.ylabel('Predicted Pt')
plt.savefig('plots/'+njet+'_'+dsid+'/xgb_test_pt_scatter.png')
