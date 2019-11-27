import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
import sys
import torch
#import h2o
#h2o.init()

#dsid = sys.argv[1]
#njet = sys.argv[2]

inFile = sys.argv[1]
outDir = sys.argv[2]

inDF = pd.read_csv(inFile)#, nrows=600000)
#if njet=='GN2':
#    inDF = inDF[inDF['is2LSS0Tau']==1]
#    inDF = inDF.drop(['is2LSS0Tau'],axis=1)

#if not inDF['comboScore'].empty:
#inDF = inDF[inDF['comboScore'] > 0.3]
#inDF = inDF.drop('comboScore', axis=1)

#inDF['higgs_pt'].values[inDF['higgs_pt']>10e6]=10e6
#inDF.loc[inDF['higgs_pt']>10e5, 'higgs_pt']=10e5

inDF = sk.utils.shuffle(inDF)

train, test = train_test_split(inDF, test_size=0.2)
'''
#Convert to h2o frames
h2o_train = h2o.H2OFrame(train)
h2o_test = h2o.H2OFrame(test)

h2o.export_file(h2o_train, '/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_train_'+outDir+'.csv', force=True)
h2o.export_file(h2o_test, '/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_test_'+outDir+'.csv', force=True)
'''
y_train = train['higgs_pt']
y_test = test['higgs_pt']

#nBin_train = train['nBin']
#nBin_test = test['nBin']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

#train = train.drop(['nBin'],axis=1)
#test = test.drop(['nBin'],axis=1)

#Convert to xgb matrices
print(list(train))
xgb_train = xgb.DMatrix(train, label=y_train, feature_names=list(train))
xgb_test = xgb.DMatrix(test, label=y_test, feature_names=list(train))

xgb_train.save_binary('tensors/xgb_train_'+outDir+'.buffer')
xgb_test.save_binary('tensors/xgb_test_'+outDir+'.buffer')

#Convert data to pytorch tensors
x_train = torch.tensor(train.values, dtype=torch.float32)
x_test = torch.tensor(test.values, dtype=torch.float32)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)
#nBin_train = torch.FloatTensor(nBin_train.values)
#nBin_test = torch.FloatTensor(nBin_test.values)

torch.save(x_train, 'tensors/torch_x_train_'+outDir+'.pt')
torch.save(y_train, 'tensors/torch_y_train_'+outDir+'.pt')
torch.save(x_test, 'tensors/torch_x_test_'+outDir+'.pt')
torch.save(y_test, 'tensors/torch_y_test_'+outDir+'.pt')

#torch.save(nBin_train, 'tensors/torch_nBin_train_'+outDir+'.pt')
#torch.save(nBin_test, 'tensors/torch_nBin_test_'+outDir+'.pt')

