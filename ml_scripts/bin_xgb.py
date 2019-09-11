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
import pickle
import sys
import torch
import scipy

outDir = sys.argv[1]
#dsid = sys.argv[1]
#njet = sys.argv[2]
'''
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
#features=['M(jjl)','Mjj','Pt(jjl)','comboScore','dR(jj, l)','dRjl0H','dRjl0O','dRjl1H','dRjl1O','dRll','dRtj00','dRtj01','dRtj10','dRtj11','dRtl0H','dRtl1H','jet_MV2c10_h0','jet_MV2c10_h1','jet_Pt_0','jet_Pt_1','lep_Pt_H','lep_Pt_O','top_Pt_0','top_Pt_1']

features=['MET', 'MET_phi', 'comboScore', 'jet_E_h0', 'jet_E_h1', 'jet_Eta_h0', 'jet_Eta_h1', 'jet_MV2c10_h0', 'jet_MV2c10_h1', 'jet_Phi_h0', 'jet_Phi_h1', 'jet_Pt_h0', 'jet_Pt_h1', 'lep_E_H', 'lep_E_O', 'lep_Eta_H', 'lep_Eta_O', 'lep_Phi_O', 'lep_Pt_H', 'lep_Pt_O', 'topScore', 'top_E_0', 'top_E_1', 'top_Eta_0', 'top_Eta_1', 'top_MV2c10_0', 'top_MV2c10_1', 'top_Phi_0', 'top_Phi_1', 'top_Pt_0', 'top_Pt_1']

if 'higgs2l' in outDir:
    features = ['MET', 'MET_phi', 'comboScore', 'lep_E_0', 'lep_E_1', 'lep_E_2', 'lep_Eta_0', 'lep_Eta_1', 'lep_Eta_2', 'lep_Phi_1', 'lep_Phi_2', 'lep_Pt_0', 'lep_Pt_1', 'lep_Pt_2', 'top_E_0', 'top_E_1', 'top_Eta_0', 'top_Eta_1', 'top_MV2c10_0', 'top_MV2c10_1', 'top_Phi_0', 'top_Phi_1', 'top_Pt_0', 'top_Pt_1']

elif 'higgs1l' in outDir:
    features = ['MET', 'MET_phi', 'comboScore', 'jet_E_h0', 'jet_E_h1', 'jet_Eta_h0', 'jet_Eta_h1', 'jet_MV2c10_h0', 'jet_MV2c10_h1', 'jet_Phi_h0', 'jet_Phi_h1', 'jet_Pt_h0', 'jet_Pt_h1', 'lep_E_0', 'lep_E_1', 'lep_E_H', 'lep_Eta_0', 'lep_Eta_1', 'lep_Eta_H', 'lep_Phi_0', 'lep_Phi_1', 'lep_Pt_0', 'lep_Pt_1', 'lep_Pt_H', 'top_E_0', 'top_E_1', 'top_Eta_0', 'top_Eta_1', 'top_MV2c10_0', 'top_MV2c10_1', 'top_Phi_0', 'top_Phi_1', 'top_Pt_0', 'top_Pt_1']


xgb_test = xgb.DMatrix('../inputData/tensors/xgb_test_'+outDir+'.buffer', feature_names = features)
xgb_train = xgb.DMatrix('../inputData/tensors/xgb_train_'+outDir+'.buffer', feature_names = features)

y_test = torch.load('../inputData/tensors/torch_y_test_'+outDir+'.pt')
y_test = y_test.float().detach().numpy()
y_train = torch.load('../inputData/tensors/torch_y_train_'+outDir+'.pt')
y_train = y_train.float().detach().numpy()

params = {
    'learning_rate' : 0.01,
    'max_depth': 9,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'eval_metric': 'auc',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=800, verbose_eval=True)

best_nrounds = pd.Series.idxmax(gbm['test-auc-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open("xgb_models/"+outDir+".dat", "wb"))

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

test_loss = sk.metrics.roc_auc_score(y_test, y_test_pred)
train_loss = sk.metrics.roc_auc_score(y_train, y_train_pred)

plt.figure()
fip = xgb.plot_importance(bst)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/xgb_feature_importance.png')


testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

plt.figure()
plt.hist(testPredTrue, 30, log=False, alpha=0.5, label='High Pt')
plt.hist(testPredFalse[:len(testPredTrue)], 30, log=False, alpha=0.5, label='Low Pt')
plt.title("BDT Output, Test Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/xgb_test_score.png')

plt.figure()
plt.hist(trainPredTrue, 30, log=False, alpha=0.5, label='High Pt')
plt.hist(trainPredFalse[:len(trainPredTrue)], 30, log=False, alpha=0.5, label='Low Pt')
plt.title("BDT Output, Train Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/xgb_train_score.png')

c = 150000

plt.figure()

yTrain = y_train#np.where(y_train > c, 1, 0)
ypTrain = y_train_pred
auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)
plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))

yTest = y_test#np.where(y_test > c, 1, 0)
ypTest = y_test_pred
auc = sk.metrics.roc_auc_score(yTest,ypTest)
fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)
plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))

plt.title("XGBoost ROC")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/xgb_roc.png')

