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
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import scipy
import h2o
from fastkde import fastKDE

alg = sys.argv[2]
model = sys.argv[3]
#dsid = sys.argv[1]
#njet = sys.argv[2]
outDir = sys.argv[1]

class Net(nn.Module):

    def __init__(self, D_in, nodes, layers):
        self.layers = layers
        super().__init__()
        self.fc1 = nn.Linear(D_in, nodes)
        self.relu1 = nn.LeakyReLU()
        self.dout = nn.Dropout(0.2)
        #self.fc2 = nn.Linear(50, 100)                                                                                                           
        self.fc = nn.Linear(nodes, nodes)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(nodes, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        h1 = self.dout(self.relu1(self.fc1(input_)))
        for i in range(self.layers):
            h1 = self.dout(self.relu1(self.fc(h1)))
        a1 = self.out(h1)
        y = self.out_act(a1)
        return y

def predict_h2o(model, outDir):
    h2o.init()

    th2o_model = h2o.load_model(model)

    th2o_train = h2o.import_file(path='/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_train_'+outDir+'.csv', destination_frame='th2o_train')
    th2o_test = h2o.import_file(path='/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_train_'+outDir+'.csv', destination_frame='th2o_test')

    y_train_pred = th2o_model.predict(th2o_train)
    y_test_pred = th2o_model.predict(th2o_test)
    
    y_train = th2o_train[3].as_data_frame().values                                
    y_test = th2o_test[3].as_data_frame().values      
    y_train_pred = y_train_pred.as_data_frame().values                                    
    y_test_pred = y_test_pred.as_data_frame().values

    y_train = y_train[:,0]
    y_train_pred = y_train_pred[:,0]
    y_test = y_test[:,0]
    y_test_pred = y_test_pred[:,0]

    plt.figure()                                                                                                                             
    plt.rcdefaults()                                                                                                                    
    fig, ax = plt.subplots()                                                                                                 
    variables = th2o_model._model_json['output']['variable_importances']['variable']                                   
    y_pos = np.arange(len(variables))                                                                  
    scaled_importance = th2o_model._model_json['output']['variable_importances']['scaled_importance']                                    
    ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')                                                      
    ax.set_yticks(y_pos)                                                                                                                 
    ax.set_yticklabels(variables)                                                                                                      
    ax.invert_yaxis()                                                                                                           
    ax.set_xlabel('Scaled Importance')                                                                                        
    ax.set_title('Variable Importance')                                                                                             
    plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+outDir+'/H2O_var_important.png')

    make_plots(y_train, y_train_pred, y_test, y_test_pred, outDir, 'H2O', '6l_450n')

def predict_xgb(model, outDir):

    bst = pickle.load(open(model, "rb"))

    xgb_test = xgb.DMatrix('../inputData/tensors/xgb_test_'+outDir+'.buffer', feature_names = bst.feature_names) 
    xgb_train = xgb.DMatrix('../inputData/tensors/xgb_train_'+outDir+'.buffer', feature_names = bst.feature_names)        
    
    y_test = torch.load('../inputData/tensors/torch_y_test_'+outDir+'.pt')             
    y_test = y_test.float().detach().numpy()                                                                
    y_train = torch.load('../inputData/tensors/torch_y_train_'+outDir+'.pt')                      
    y_train = y_train.float().detach().numpy()

    y_test_pred = bst.predict(xgb_test)
    y_train_pred = bst.predict(xgb_train)

    plt.figure()
    fip = xgb.plot_importance(bst)
    plt.title("XGBoost feature important")
    plt.legend(loc='lower right')
    plt.savefig('plots/'+outDir+'/XGBoost_feature_importance.png')

    torch.save(y_test_pred, 'xgb_models/'+outDir+'_y_test_pred.pt')
    torch.save(y_train_pred, 'xgb_models/'+outDir+'_y_train_pred.pt')

    make_plots(y_train, y_train_pred, y_test, y_test_pred, outDir, 'XGBoost', '')

def predict_torch(model, outDir):

    X = torch.load('../inputData/tensors/torch_x_train_'+outDir+'.pt')
    X_test = torch.load('../inputData/tensors/torch_x_test_'+outDir+'.pt')

    Y = torch.load('../inputData/tensors/torch_y_train_'+outDir+'.pt')
    Y_test = torch.load('../inputData/tensors/torch_y_test_'+outDir+'.pt')

    net = Net(X.shape[1],600,4)
    net.load_state_dict(torch.load(model))
    net.eval()

    def normalize(x):
        x_normed = x / x.max(0, keepdim=True)[0]
        return x_normed

    x_train = normalize(X)
    y_train = normalize(Y)
        
    x_test = normalize(X_test)
    y_test = normalize(Y_test)

    y_train_pred = net(X)[:,0]
    y_train_pred = y_train_pred.float().detach().numpy()

    y_test_pred = net(X_test)[:,0]
    y_test_pred = y_test_pred.float().detach().numpy()

    y_train = y_train*Y.max()
    y_train_pred = y_train_pred*Y.max()

    y_test = y_test*Y_test.max()
    y_test_pred = y_test_pred*Y_test.max()

    make_plots(y_train, y_train_pred, y_test, y_test_pred, outDir, 'Torch', '')

def make_plots(y_train, y_train_pred, y_test, y_test_pred, outDir, alg, conf):

    train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)
    test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)

    #train pt histogram
    plt.figure()
    plt.hist(y_train, 20, log=False, range=(0,800000), alpha=0.5, label='truth')
    plt.hist(y_train_pred, 20, log=False, range=(0,800000), alpha=0.5, label='train')
    plt.title(alg+" Train Data, mae=%0.f" %(train_loss))
    plt.xlabel('Higgs Pt')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+outDir+'/'+alg+'_train_pt_spectrum_'+conf+'n.png')

    #test pt histogram
    plt.figure()
    plt.hist(y_test, 20, log=False, range=(0,800000), alpha=0.5, label='truth')
    plt.hist(y_test_pred, 20, log=False, range=(0,800000), alpha=0.5, label='test')
    plt.title(alg+" Test Data, mae=%0.f" %(test_loss))
    plt.xlabel('Higgs Pt')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+outDir+'/'+alg+'_test_pt_spectrum_'+conf+'.png')

    # train scatter plot
    print(y_train)
    print(y_train_pred)
    xy_train = np.vstack([y_train, y_train_pred])
    z_train = scipy.stats.gaussian_kde(xy_train)(xy_train)

    plt.figure()
    plt.scatter(y_train, y_train_pred, c=np.log(z_train), edgecolor='')
    plt.title(alg+" Test Data, mae=%0.f" %(test_loss))
    plt.xlabel('Truth Pt')
    plt.ylabel('Predicted Pt')
    plt.plot([0,0.6],[0,0.6], zorder=10)
    plt.savefig('plots/'+outDir+'/'+alg+'_train_pt_scatter_'+conf+'.png')

    # test scatter plot                                                                                                                         
    print(y_test.shape)
    xy_test = np.vstack([y_test, y_test_pred])
    z_test = scipy.stats.gaussian_kde(xy_test)(xy_test)

    plt.figure()
    plt.scatter(y_test, y_test_pred, c=np.log(z_test), edgecolor='')
    plt.title(alg+" Test Data, mae=%0.f" %(test_loss))
    plt.xlabel('Truth Pt')
    plt.ylabel('Predicted Pt')
    plt.plot([0,1200000],[0,1200000], zorder=10)
    plt.savefig('plots/'+outDir+'/'+alg+'_test_pt_scatter_'+conf+'.png')

    cutoff = [150000, 200000, 250000]
    #if alg=='torch': cutoff = [0.15, 0.20, 0.25]

    #Train ROC
    plt.figure()
    for c in cutoff:
        yTrain = np.where(y_train > c, 1, 0)
        ypTrain = y_train_pred

        auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
        fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)

        plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

        plt.title(alg+" Train ROC")
        plt.legend(loc='lower right')
    plt.savefig('plots/'+outDir+'/'+alg+'_train_roc_'+conf+'.png')

    #Test ROC                                                                                                                                  
    plt.figure()
    for c in cutoff:
        yTest = np.where(y_test > c, 1, 0)
        ypTest = y_test_pred

        auc = sk.metrics.roc_auc_score(yTest,ypTest)
        fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

        plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

        plt.title(alg+" Test ROC")
        plt.legend(loc='lower right')
    plt.savefig('plots/'+outDir+'/'+alg+'_test_roc_'+conf+'.png')

            
if alg=='Torch': predict_torch(model, outDir)
elif alg=='XGBoost': predict_xgb(model,outDir)
elif alg=='H2O': predict_h2o(model, outDir)
else: print('no such model: '+alg)
