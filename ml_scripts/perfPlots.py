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

def make_plots(y_train, y_train_pred, y_test, y_test_pred, njet, dsid, model, conf):

    train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)
    test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)

    #train pt histogram
    plt.figure()
    plt.hist(y_train, 20, log=False, range=(0,0.8), alpha=0.5, label='truth')
    plt.hist(y_train_pred, 20, log=False, range=(0,0.8), alpha=0.5, label='train')
    plt.title(model+" Train Data, mae=%0.4f" %(train_loss))
    plt.xlabel('Higgs Pt')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+njet+'_'+dsid+'/'+model+'_train_pt_spectrum_'+conf+'n.png')

    #test pt histogram
    plt.figure()
    plt.hist(y_test, 20, log=False, range=(0,0.8), alpha=0.5, label='truth')
    plt.hist(y_test_pred, 20, log=False, range=(0,0.8), alpha=0.5, label='test')
    plt.title(model+" Test Data, mae=%0.4f" %(test_loss))
    plt.xlabel('Higgs Pt')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+njet+'_'+dsid+'/'+model+'_test_pt_spectrum_'+conf+'.png')

    # train scatter plot
    xy_train = np.vstack([y_train, y_train_pred])
    z_train = scipy.stats.gaussian_kde(xy_train)(xy_train)

    plt.figure()
    plt.scatter(y_train, y_test_pred, c=z_train, edgecolor='')
    plt.title(model+" Test Data, mae=%0.4f" %(test_loss))
    plt.xlabel('Truth Pt')
    plt.ylabel('Predicted Pt')
    plt.plot([0,0.6],[0,0.6], zorder=10)
    plt.savefig('plots/'+njet+'_'+dsid+'/'+model+'_train_pt_scatter_'+conf+'.png')

    # test scatter plot                                                                                                                         
    xy_test = np.vstack([y_test, y_test_pred])
    z_test = scipy.stats.gaussian_kde(xy_test)(xy_test)

    plt.figure()
    plt.scatter(y_test, y_test_pred, c=z_test, edgecolor='')
    plt.title(model+" Test Data, mae=%0.4f" %(test_loss))
    plt.xlabel('Truth Pt')
    plt.ylabel('Predicted Pt')
    plt.plot([0,0.6],[0,0.6], zorder=10)
    plt.savefig('plots/'+njet+'_'+dsid+'/'+model+'_test_pt_scatter_'+conf+'.png')

    cutoff = [150000, 200000, 250000]
    if model=='torch': cutoff = [0.15, 0.20, 0.25]

    #Train ROC
    plt.figure()
    for c in cutoff:
        yTrain = np.where(y_train > c, 1, 0)
        ypTrain = y_train_pred

        auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
        fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)

        plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

        plt.title(model+" Train ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
        plt.legend(loc='lower right')
    plt.savefig('plots/'+njet+'_'+dsid+'/'+model+'_train_roc_'+conf+'.png')

    #Test ROC                                                                                                                                  
    plt.figure()
    for c in cutoff:
        yTest = np.where(y_test > c, 1, 0)
        ypTest = y_test_pred

        auc = sk.metrics.roc_auc_score(yTest,ypTest)
        fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

        plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

        plt.title(model+" Test ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
        plt.legend(loc='lower right')
    plt.savefig('plots/'+njet+'_'+dsid+'/'+model+'_test_roc_'+conf+'.png')

        
