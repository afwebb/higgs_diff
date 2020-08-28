import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras import losses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import scipy


def makePlots(alg, model, outDir, y_train, y_test, y_train_pred, y_test_pred):

    if 'bin' in alg:
        binned = True
    else:
        binned = False

    test_loss = np.sqrt(sk.metrics.mean_squared_error(y_test, y_test_pred))
    train_loss = np.sqrt(sk.metrics.mean_squared_error(y_train, y_train_pred))

    #ROC curve
    plt.figure()
    c = 150000
    if not binned:
        c = 150000
        yTrain = np.where(y_train > c, 1, 0)
        yTest = np.where(y_test > c, 1, 0)

    ypTrain = y_train_pred
    auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
    fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)
    plt.plot(fpr, tpr, label='train AUC = %.3f' %(auc))
    
    ypTest = y_test_pred
    auc = sk.metrics.roc_auc_score(yTest,ypTest)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)
    plt.plot(fpr, tpr, label='test AUC = %.3f' %(auc))
    
    plt.title("Keras ROC")
    plt.legend(loc='lower right')
    plt.savefig(f'plots/{outDir}/{alg}_roc.png')
    
    if not binned:
        #Pt scatter density plot

        # Calculate the point density                                                                            
        scatterSize = 60000
        xy = np.vstack([y_test[:scatterSize], y_test_pred[:scatterSize]])
        z = scipy.stats.gaussian_kde(xy)(xy)
        
        plt.figure()
        plt.scatter(y_test[:scatterSize]/1000, y_test_pred[:scatterSize]/1000, c=np.log(z), edgecolor='')
        plt.title("Keras Test Data, MSE=%f" %(test_loss))
        plt.xlabel('Truth $p_T$ [GeV]')
        plt.ylabel('Predicted $p_T$ [GeV]')
        plt.xlim(1,1000)
        plt.ylim(1,1000)
        plt.plot([0,1000],[0,1000],zorder=10)
        plt.savefig(f'plots/{outDir}/{alg}_test_pt_scatter.png')
        
        # Calculate the point density                                                                                    
        xy = np.vstack([y_train[:scatterSize], y_train_pred[:scatterSize]])
        z = scipy.stats.gaussian_kde(xy)(xy)
        
        plt.figure()
        plt.scatter(y_train[:scatterSize]/1000, y_train_pred[:scatterSize]/1000, c=np.log(z), edgecolor='')
        plt.title("Keras Train Data, MSE=%f" %(train_loss))
        plt.xlabel('Truth $p_T$ [GeV]')
        plt.ylabel('Predicted $p_T$ [GeV]')
        plt.xlim(0,1000)
        plt.ylim(0,1000)
        plt.plot([0,1000],[0,1000],zorder=10)
        plt.savefig(f'plots/{outDir}/{alg}_train_pt_scatter.png')

        errBins = np.linspace(0, 1200000, num=30)
        testErrVec = np.array(np.sqrt((y_test_pred - y_test)**2))
        print(testErrVec)
        y_pred_bins = []
        y_err_bins = []
        
        for b in range(1, len(errBins)):
            binned_y = testErrVec[(y_test<errBins[b]) & (y_test>errBins[b-1])]
            y_pred_bins.append( np.mean( binned_y ) )
            y_err_bins.append(  np.std(binned_y) )
            
        binned_y = testErrVec[(y_test>1200000)]
        y_pred_bins.append( np.mean( binned_y ) )
        y_err_bins.append(  np.std(binned_y) )
        errBins = np.append( errBins, np.mean(y_test[y_test>1200000]) )
        
        print(len(errBins), len(y_pred_bins), len(y_err_bins))
        
        errBins = np.divide(errBins, 1000)
        y_pred_bins = np.divide(y_pred_bins, 1000)
        y_err_bins = np.divide(y_err_bins, 1000)
        
        plt.figure()
        plt.errorbar( errBins[1:], y_pred_bins, yerr=y_err_bins)
        plt.plot([0,1200],[0,1200])#, zorder=10)
        plt.title('Prediction Error')
        plt.xlabel('Truth Higgs Pt [MeV]')
        plt.ylabel('RMSE')
        plt.savefig(f'plots/{outDir}/{alg}_err.png')

        #Histogram of high/low pt predictions
        y_test_pred_high = y_test_pred[(y_test>150000)]
        y_test_pred_low = y_test_pred[(y_test>150000)]
        
        y_train_pred_high = y_train_pred[(y_train>150000)]
        y_train_pred_low = y_train_pred[(y_train>150000)]
        
        nHL = min([len(y_test_pred_high), len(y_test_pred_low), len(y_train_pred_high), len(y_train_pred_low)])
        
        plt.figure()
        plt.hist(y_test_pred_high[:nHL]/1000, 30, range=(0, 450), log=False, alpha=0.5, label='High Pt - Test')
        plt.hist(y_test_pred_low[:nHL]/1000, 30, range=(0, 450), log=False, alpha=0.5, label='Low Pt - Test')
        plt.hist(y_train_pred_high[:nHL]/1000, 30, range=(0, 450), log=False, histtype='step', alpha=0.5, label='High Pt - Train')
        plt.hist(y_train_pred_low[:nHL]/1000, 30, range=(0, 450), log=False, histtype='step', alpha=0.5, label='Low Pt - Train')
        plt.title("Keras Output")
        plt.xlabel('Predicted Higgs $p_T$ [GeV]')
        plt.ylabel('NEvents')
        plt.legend(loc='upper right')                                                                  
        plt.savefig(f'plots/{outDir}/{alg}_score.png')
    
    if binned:
        #Histogram of scores
        testPredTrue = y_test_pred[y_test==1]
        testPredFalse = y_test_pred[y_test==0]
        
        trainPredTrue = y_train_pred[y_train==1]
        trainPredFalse = y_train_pred[y_train==0]
        
        plt.figure()
        plt.hist(testPredTrue, 30, range=(-0.1,1.1), log=False, alpha=0.5, label='High Pt - Test')
        plt.hist(testPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, alpha=0.5, label='Low Pt - Test')
        plt.hist(trainPredTrue[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='High Pt - Train')
        plt.hist(trainPredFalse[:len(testPredTrue)], 30, range=(-0.1,1.1), log=False, histtype='step', alpha=0.5, label='Low Pt - Train')
        plt.title("Keras Output")
        plt.xlabel('Keras Score')
        plt.ylabel('NEvents')
        plt.legend(loc='upper right')
        plt.savefig(f'plots/{outDir}/{alg}_score.png')
