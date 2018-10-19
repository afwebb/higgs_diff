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

dsid = sys.argv[1]
njet = sys.argv[2]

#inDF = pd.read_csv('../inputData/345673_4j.csv')
inDF = pd.read_csv('../inputData/'+dsid+'_'+njet+'j.csv')

cutoff = 300000

dsid = sys.argv[1]
njet = sys.argv[2]

#inDF = inDF[inDF['is2LSS0Tau']==0]

inDF.loc[ inDF['higgs_pt'] <= cutoff, 'higgs_pt'] = 0 ;
inDF.loc[ inDF['higgs_pt'] > cutoff, 'higgs_pt'] = 1 ;

train, test = train_test_split(inDF, test_size=0.3)

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

xgb_train = xgb.DMatrix(train, label=y_train)
xgb_test = xgb.DMatrix(test, label=y_test)

params = {
    'learning_rate' : 0.0001,
    'max_depth': 25,
    'min_child_weight': 2,
    'gamma': 0.9,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'objective': 'binary:logistic',
    'nthread': -1,
    'scale_pos_weight':1
}

gbm = xgb.cv(params, xgb_train, num_boost_round=200, verbose_eval=True)

best_nrounds = pd.Series.idxmin(gbm['test-error-mean'])
print( best_nrounds)

bst = xgb.train(params, xgb_train, num_boost_round=best_nrounds, verbose_eval=True)
pickle.dump(bst, open("xgb_models/"+njet+'j_'+dsid+".dat", "wb"))

y_pred = bst.predict(xgb_test)

auc = sk.metrics.roc_auc_score(y_test,y_pred)
print( auc)

roc_array = sk.metrics.roc_curve(y_test,y_pred)

plt.figure()
plt.plot(roc_array[1], label='AUC = '+str(auc))
plt.title("xgboost ROC, nround ="+str(best_nrounds))
plt.legend(loc='lower right')
plt.savefig('plots/'+njet+'j_'+dsid+'/xgb_roc_curve.png', label='AUC = '+str(auc))                                                                          

#y_pred[:,1]                                                                                                                                     
plt.figure()
fip = xgb.plot_importance(bst)
plt.title("xgboost feature important")
plt.legend(loc='lower right')
plt.savefig('plots/'+njet+'j_'+dsid+'/xgb_feature_importance.png')

cm = confusion_matrix(y_test, (y_pred>0.5))
print( cm)
