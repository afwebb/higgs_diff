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

inFile = sys.argv[1]
outDir = sys.argv[2]
#dsid = sys.argv[1]
#njet = sys.argv[2]

#inDF = pd.read_csv('../inputData/345673_4j.csv')

inDF = pd.read_csv(inFile)
inDF = inDF.dropna()

if outDir=="2lSS_highPt":
    inDF = inDF[1.4*inDF['recoHiggsPt_2lSS']-40e3>150000]
    #inDF = inDF.drop(['recoHiggsPt_2lSS'],axis=1)
    #inDF = inDF.drop(['bin_score'],axis=1)
elif outDir=='2lSS_lowPt':
    inDF = inDF[1.4*inDF['recoHiggsPt_2lSS']-40e3<150000]
    #inDF = inDF.drop(['recoHiggsPt_2lSS'],axis=1)
    #inDF = inDF.drop(['bin_score'],axis=1)

if outDir=="3lF":
    inDF = inDF[inDF['decayScore']>0.18]
    inDF = inDF.drop(['recoHiggsPt_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)
elif outDir=="3lS":
    inDF = inDF[inDF['decayScore']<0.18]
    inDF = inDF.drop(['recoHiggsPt_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)

if outDir=="3lF_highPt":
    inDF = inDF[inDF['decayScore']>0.18]
    inDF = inDF[inDF['recoHiggsPt_3lF']>150000]
    #inDF = inDF.drop(['recoHiggsPt_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)
    inDF = inDF.drop(['recoHiggsPt_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)
elif outDir=="3lS_highPt":
    inDF = inDF[inDF['decayScore']<0.18]
    inDF = inDF[1.2*inDF['recoHiggsPt_3lS']-20e3>150000]
    inDF = inDF.drop(['recoHiggsPt_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)
    #inDF = inDF.drop(['recoHiggsPt_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)

if outDir=="3lF_lowPt":
    inDF = inDF[inDF['decayScore']>0.18]
    inDF = inDF[inDF['recoHiggsPt_3lF']<150000]
    #inDF = inDF.drop(['recoHiggsPt_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)
    inDF = inDF.drop(['recoHiggsPt_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)
elif outDir=="3lS_lowPt":
    inDF = inDF[inDF['decayScore']<0.18]
    inDF = inDF[1.2*inDF['recoHiggsPt_3lS']-20e3<150000]
    inDF = inDF.drop(['recoHiggsPt_3lF'],axis=1)
    #inDF = inDF.drop(['bin_score_3lF'],axis=1)
    #inDF = inDF.drop(['recoHiggsPt_3lS'],axis=1)
    #inDF = inDF.drop(['bin_score_3lS'],axis=1)


print(inDF.isnull().sum())

#inDF = inDF[inDF['dNN_bin_score']>0.4]
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

weight_train = train['weight']
weight_test = test['weight']
train = train.drop(['weight'], axis=1)
test = test.drop(['weight'], axis=1)

xgb_train = xgb.DMatrix(train, label=y_train, feature_names=list(test), weight=weight_train)
xgb_test = xgb.DMatrix(test, label=y_test, feature_names=list(train), weight=weight_test)
'''
features=['DRlj00', 'DRll01', 'HT', 'MET_RefFinal_et', 'MET_RefFinal_phi', 'Mll01', 'Ptll01', 'dNN_bin_score', 'dNN_pt_score', 'dilep_type', 'lead_jetEta', 'lead_jetPhi', 'lead_jetPt', 'lep_Eta_0', 'lep_Eta_1', 'lep_Phi_0', 'lep_Phi_1', 'lep_Pt_0', 'lep_Pt_1', 'nJets_OR_T', 'nJets_OR_T_MV2c10_70', 'sublead_jetEta', 'sublead_jetPhi', 'sublead_jetPt']

xgb_test = xgb.DMatrix('xgb_test_'+outDir+'.buffer', feature_names = features)
xgb_train = xgb.DMatrix('xgb_train_'+outDir+'.buffer', feature_names = features)

y_test = torch.load('torch_y_test_'+outDir+'.pt')
y_test = y_test.float().detach().numpy()
y_train = torch.load('torch_y_train_'+outDir+'.pt')
y_train = y_train.float().detach().numpy()
'''
params = {
    'learning_rate' : 0.005,
    'max_depth': 7,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'eval_metric': 'auc',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=500, verbose_eval=True)

best_nrounds = pd.Series.idxmax(gbm['test-auc-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open("models/"+outDir.replace('/','_')+".dat", "wb"))

y_test_pred = bst.predict(xgb_test)
y_train_pred = bst.predict(xgb_train)

test_loss = sk.metrics.roc_auc_score(y_test, y_test_pred)
train_loss = sk.metrics.roc_auc_score(y_train, y_train_pred)

make_plots('xgb', bst, outDir, y_train, y_test, y_train_pred, y_test_pred)
'''
plt.figure()
fip = xgb.plot_importance(bst)
f = fip.figure
f.set_size_inches(14, 8)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/'+outDir+'/kerasIn_xgb_feature_importance.png')


testPredTrue = y_test_pred[y_test==1]
testPredFalse = y_test_pred[y_test==0]

trainPredTrue = y_train_pred[y_train==1]
trainPredFalse = y_train_pred[y_train==0]

plt.figure()
plt.hist(testPredTrue, 30, log=False, alpha=0.5, label='Signal')
plt.hist(testPredFalse[:len(testPredTrue)], 30, log=False, alpha=0.5, label='Background')
plt.title("BDT Output, Test Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/kerasIn_xgb_test_score.png')

plt.figure()
plt.hist(trainPredTrue, 30, log=False, alpha=0.5, label='Signal')
plt.hist(trainPredFalse[:len(trainPredTrue)], 30, log=False, alpha=0.5, label='Background')
plt.title("BDT Output, Train Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/kerasIn_xgb_train_score.png')

plt.figure()
plt.hist(testPredTrue, 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Signal - Test')
plt.hist(testPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Background - Test')
plt.hist(trainPredTrue[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Signal - Train')
plt.hist(trainPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Background - Train')
plt.title("BDT Output, Train Data")
plt.xlabel('BDT Score')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+outDir+'/xgb_score.png')

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


f = plt.figure(figsize=(19,15))
plt.matshow(inDF.corr(),fignum=f.number)
for (i,j), z in np.ndenumerate(inDF.corr()):
    plt.text(j,i, '{:0.2f}'.format(z), ha='center', va='center')
plt.xticks(range(inDF.shape[1]), inDF.columns, fontsize=14, rotation=45)
plt.yticks(range(inDF.shape[1]), inDF.columns, fontsize=14)
cb = plt.colorbar()
plt.savefig('plots/'+outDir+"/kerasIn_CorrMat.png")
plt.close()
'''
