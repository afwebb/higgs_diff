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

def cat_plots(alg, model, outDir, y_train, y_test, y_train_pred, y_test_pred):
    '''
    Given the name of the training algorithm, the truth and predicted y values, plots the performance of the algorithm
    Plots include a histogram of the MVA score, ROC curve, and feature importance for xgb
    '''

    test_loss = sk.metrics.log_loss(y_test, y_test_pred)
    train_loss = sk.metrics.log_loss(y_train, y_train_pred)

    testPredTrue = y_test_pred[y_test==1]
    testPredFalse = y_test_pred[y_test==0]
    
    trainPredTrue = y_train_pred[y_train==1]
    trainPredFalse = y_train_pred[y_train==0]

    # Histogram of MVA output score
    minLen = min([len(testPredTrue), len(testPredFalse), len(trainPredTrue), len(trainPredFalse)])
    plt.figure()
    plt.hist(testPredTrue[:minLen], 30, range=(-0.1,0.7), log=False, alpha=0.5, label='Correct - Test')
    plt.hist(testPredFalse[:minLen], 30, range=(-0.1,0.7), log=False, alpha=0.5, label='Incorrect - Test')
    plt.hist(trainPredTrue[:minLen], 30, range=(-0.1,0.7), log=False, histtype='step', alpha=0.5, label='Correct - Train')
    plt.hist(trainPredFalse[:minLen], 30, range=(-0.1,0.7), log=False, histtype='step', alpha=0.5, label='Incorrect - Train')
    plt.title(f"{alg} Output")
    plt.xlabel(f'{alg} Score')
    plt.ylabel('NEvents')
    plt.legend()
    plt.savefig(f'plots/{outDir}/{alg}_score.png')

    y_test_pred_best = [sorted(x.argsort()[-2:]) for x in y_test_pred]
    y_test_pred_cat = np.zeros(y_test.shape)

    for yp, b in zip(y_test_pred_cat, y_test_pred_best):
        yp[b] = 1

    print(y_test_pred_cat, y_test)
    confMat = sklearn.metrics.multilabel_confusion_matrix(y_test_pred_cat, y_test)
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(sum(confMat)/len(y_test), annot=True, robust=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(f"{alg.capitalize()} Confusion Matrix")
    plt.savefig(f'plots/{outDir}/{alg}_conf_matrix.png')

    #Loss history - keras 
    if alg == 'keras':
        plt.figure()
        plt.plot(model.history['loss'], label='Train Loss')
        #plt.plot(model.history['val_loss'], label='Test Loss')
        plt.title(f"{alg} Loss")
        plt.xlabel('Epoch')
        plt.ylabel('BCE')                                                                                                
        plt.legend()
        plt.savefig(f'plots/{outDir}/keras_BCE_history.png')
        
        #plt.figure()
        #plt.plot(model.history['AUC'], label='Train AUC')
        ##plt.plot(model.history['val_AUC'], label='Test AUC')
        #plt.title("Keras AUC")
        #plt.xlabel('Epoch')                                                                                         
        #plt.ylabel('AUC')                                                                         
        #plt.legend()
        #plt.savefig(f'plots/{outDir}/keras_auc_history.png')

    #Feature importance - XGB
    if alg=='xgb':
        plt.figure()
        fip = xgb.plot_importance(model)
        plt.title("xgboost feature important")
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
        plt.title('XGBoost AUC')
        plt.savefig('plots/'+outDir+'/xgb_auc_history.png')
