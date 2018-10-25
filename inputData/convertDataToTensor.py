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

dsid = sys.argv[1]
njet = sys.argv[2]

inDF = pd.read_csv('../inputData/'+dsid+'_'+njet+'.csv')
if njet=='GN2':
    inDF = inDF[inDF['is2LSS0Tau']==1]

train, test = train_test_split(inDF, test_size=0.3)

#Convert to h2o frames
'''
h2o_train = h2o.H2OFrame(train)
h2o_test = h2o.H2OFrame(test)

h2o.export_file(h2o_train, 'tensors/h2o_train'+dsid+'_'+njet+'.csv')
h2o.export_file(h2o_test, 'tensors/h2o_test'+dsid+'_'+njet+'.csv')
'''

y_train = train['higgs_pt']
y_test = test['higgs_pt']

train = train.drop(['higgs_pt'],axis=1)
test = test.drop(['higgs_pt'],axis=1)

#Convert to xgb matrices
xgb_train = xgb.DMatrix(train, label=y_train)
xgb_test = xgb.DMatrix(test, label=y_test)

xgb_train.save_binary('tensors/xgb_train_'+dsid+'_'+njet+'.buffer')
xgb_test.save_binary('tensors/xgb_test'+dsid+'_'+njet+'.buffer')

#Convert data to pytorch tensors
x_train = torch.tensor(train.values, dtype=torch.float32)
x_test = torch.tensor(test.values, dtype=torch.float32)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)

torch.save(x_train, 'tensors/torch_x_train'+dsid+'_'+njet+'.pt')
torch.save(y_train, 'tensors/torch_y_train'+dsid+'_'+njet+'.pt')
torch.save(x_test, 'tensors/torch_x_test'+dsid+'_'+njet+'.pt')
torch.save(y_test, 'tensors/torch_y_test'+dsid+'_'+njet+'.pt')

