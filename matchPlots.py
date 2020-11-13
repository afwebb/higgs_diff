#Plot the performance of truth matching algorithms

import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
import sys
import torch
import scipy

def make_plots(alg, model, outDir, y_train, y_test, y_train_pred, y_test_pred, weights_train, weights_test):
    '''
    Given the name of the training algorithm, the truth and predicted y values, plots the performance of the algorithm
    Plots include a histogram of the MVA score, ROC curve, and feature importance for xgb
    '''

    test_loss = sk.metrics.mean_absolute_error(y_test, y_test_pred)
    train_loss = sk.metrics.mean_absolute_error(y_train, y_train_pred)

    testPredTrue = y_test_pred[y_test==1]
    testPredFalse = y_test_pred[y_test==0]
    trainPredTrue = y_train_pred[y_train==1]
    trainPredFalse = y_train_pred[y_train==0]

    wtestPredTrue = weights_test[y_test==1]
    wtestPredFalse = weights_test[y_test==0]
    wtrainPredTrue = weights_train[y_train==1]
    wtrainPredFalse = weights_train[y_train==0]

    # Histogram of MVA output score
    minLen = min([len(testPredTrue), len(testPredFalse), len(trainPredTrue), len(trainPredFalse)])
    plt.figure()
    plt.hist(testPredTrue, 30, weights=wtestPredTrue, range=(-0.1,1.1), log=False, 
             alpha=0.5, density=True,  label='Signal - Test')
    plt.hist(testPredFalse, 30, weights=wtestPredFalse, range=(-0.1,1.1), 
             log=False, alpha=0.5, density=True,  label='Background - Test')
    plt.hist(trainPredTrue, 30, weights=wtrainPredTrue,
             range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, density=True,  label='Signal - Train')
    plt.hist(trainPredFalse, 30, weights=wtrainPredFalse, 
             range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, density=True,  label='Background - Train')
    #plt.title(f"{alg} Output")
    plt.xlabel(f'{alg} Score')
    plt.ylabel('NEvents')
    plt.legend()
    plt.savefig(f'plots/{outDir}/{alg}_score.png')

    # ROC Curve
    plt.figure()
    auc = sk.metrics.roc_auc_score(y_test, y_test_pred, sample_weight=weights_test)
    fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred, sample_weight=weights_test)
    plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))
    
    auc = sk.metrics.roc_auc_score(y_train, y_train_pred, sample_weight=weights_train)
    fpr, tpr, _ = sk.metrics.roc_curve(y_train, y_train_pred, sample_weight=weights_train)
    plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))
    plt.legend()
    #plt.title(f'{alg.capitalize()} Match ROC')
    plt.xlabel('Signal Rejection Rate')
    plt.ylabel('Background Rejection Rate')
    plt.savefig(f'plots/{outDir}/{alg}_roc.png')
    
    small_pred = np.concatenate((testPredTrue[:minLen], testPredFalse[:minLen]))
    print(testPredTrue[:minLen])
    print(len(testPredFalse[:minLen]), testPredFalse[:minLen])
    print(len(small_pred), small_pred)
    small_test = [1 for x in range(minLen)]+[0 for x in range(minLen)]
    if '3lF' in outDir or 'W' in outDir: y_test_bin = np.where(small_pred > 0.5, 1, 0)
    else: y_test_bin = np.where(small_pred > 0.2, 1, 0) 
    confMat = sklearn.metrics.confusion_matrix(small_test, y_test_bin, sample_weight=weights_test[:2*minLen])/(len(small_test))
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(confMat, annot=True, robust=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    #ax.set_title(f"{alg.capitalize()} Confusion Matrix")
    plt.savefig(f'plots/{outDir}/{alg}_conf_matrix.png')

    #Loss history - keras 
    if alg == 'keras':
        plt.figure()
        plt.plot(model.history['loss'], label='Train Loss')
        #plt.plot(model.history['val_loss'], label='Test Loss')
        #plt.title(f"{alg} Loss")
        plt.xlabel('Epoch')
        plt.ylabel('BCE')                                                                                                
        plt.legend()
        plt.savefig(f'plots/{outDir}/keras_BCE_history.png')
        
        plt.figure()
        plt.plot(model.history['AUC'], label='Train AUC')
        #plt.plot(model.history['val_AUC'], label='Test AUC')
        #plt.title("Keras AUC")
        plt.xlabel('Epoch')                                                                                                  
        plt.ylabel('AUC')                                                                         
        plt.legend()
        plt.savefig(f'plots/{outDir}/keras_auc_history.png')

    #Feature importance - XGB
    if alg=='xgb':
        plt.figure()
        fip = xgb.plot_importance(model)
        #plt.title("xgboost feature important")
        plt.legend(loc='lower right')
        plt.savefig('plots/'+outDir+'/xgb_feature_importance.png')
        
        #https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
        results = model.elavs_result()
        plt.figure()
        plt.plot(x_axis, results['validation_0']['auc'], label='Train')
        plt.plot(x_axis, results['validation_1']['auc'], label='Test')
        plt.legend()
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        #plt.title('XGBoost AUC')
        plt.savefig('plots/'+outDir+'/xgb_auc_history.png')
