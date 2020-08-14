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
#from sklearn.metrics import plot_confusion_matrix
import pickle
import sys
import torch
import scipy

def make_plots(alg, outDir, y_train, y_test, y_train_pred, y_test_pred):
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
    
    # Histogram of MVA output score
    minLen = min([len(testPredTrue), len(testPredFalse), len(trainPredTrue), len(trainPredFalse)])
    plt.figure()
    plt.hist(testPredTrue[:minLen], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Correct - Test')
    plt.hist(testPredFalse[:minLen], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Incorrect - Test')
    plt.hist(trainPredTrue[:minLen], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Correct - Train')
    plt.hist(trainPredFalse[:minLen], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Incorrect - Train')
    plt.title(f"{alg} Output")
    plt.xlabel(f'{alg} Score')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig(f'plots/{outDir}/{alg}_score.png')

    #Feature importance plot - XGBoost only
    '''
    if alg=='xgb':
        plt.figure()
        fip = xgb.plot_importance(model)
        plt.title(f"{alg.capitalize()} feature important")
        plt.legend(loc='lower right')
        plt.savefig(f'plots/{outDir}/{alg}_feature_importance.png')
    '''

    # ROC Curve
    plt.figure()
    auc = sk.metrics.roc_auc_score(y_test, y_test_pred)
    fpr, tpr, _ = sk.metrics.roc_curve(y_test, y_test_pred)
    plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))
    
    auc = sk.metrics.roc_auc_score(y_train, y_train_pred)
    fpr, tpr, _ = sk.metrics.roc_curve(y_train, y_train_pred)
    plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))
    plt.legend(loc='lower right')
    plt.title(f'{alg.capitalize()} Match ROC')
    plt.savefig(f'plots/{outDir}/{alg}_roc.png')
    
    small_pred = np.concatenate((testPredTrue[:minLen], testPredFalse[:minLen]))
    print(testPredTrue[:minLen])
    print(len(testPredFalse[:minLen]), testPredFalse[:minLen])
    print(len(small_pred), small_pred)
    small_test = [1 for x in range(minLen)]+[0 for x in range(minLen)]
    #y_test_bin = np.where(y_test_pred > 0.5, 1, 0)
    y_test_bin = np.where(small_pred > 0.2, 1, 0) 
    print(len(small_test), minLen)
    confMat = sklearn.metrics.confusion_matrix(small_test, y_test_bin)/(len(small_test))
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(confMat, annot=True, robust=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(f"{alg.capitalize()} Confusion Matrix")
    plt.savefig(f'plots/{outDir}/{alg}_conf_matrix.png')

