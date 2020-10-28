'''
Take an input root file, and add a branch with the prediction of each NN
These include the top reconstruction, higgs reconstruction, pt prediction, and (for 3l) decay mode prediction algorithms
Usage: 
   python3.6 addToRoot.py <listOfFiles.txt>
'''

import pandas as pd
import tensorflow.keras
from tensorflow.keras.models import load_model
import ROOT
from ROOT import TFile
from sklearn.utils import shuffle
from rootpy.io import root_open
import root_numpy
import numpy as np
import sys
import random
from functionsMatch import jetCombosTop, findBestTopKeras, findBestHiggsTop
from dictPt import ptDictHiggsTop2lSS, ptDictHiggsTop3lF, ptDictHiggsTop3lS
from dictDecay import decayDict
from joblib import Parallel, delayed
import multiprocessing

def runReco(inf):
    #Set the channel, load in the models and normalization factors
    if '3l' in inf:
        channel='3l'
        ptDict = ptDictHiggsTop3lS

        topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy", allow_pickle=True)

        model3lF = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop3lF.h5")
        normFactors3lF = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop3lF_normFactors.npy", 
                                 allow_pickle=True)

        model3lS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop3lS.h5")
        normFactors3lS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop3lS_normFactors.npy", 
                                 allow_pickle=True)

    elif '2lSS' in inf:
        channel='2lSS'
        ptDict = ptDictHiggsTop2lSS

        topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top2lSS.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top2lSS_normFactors.npy", 
                                 allow_pickle=True)
        
        model2lSS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop2lSS.h5")
        normFactors2lSS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop2lSS_normFactors.npy", 
                                  allow_pickle=True)

    else:
        print(f'Channel {channel} is invalid. Should be 2lSS or 3l')
        return
        
    #Open the root file
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    if hasattr(nom, "recoHiggsPt_2lSS") or hasattr(nom, "recoHiggsPt_3lS"):
        print(f'{inf} already has score')
        return

    #initialize output dicts
    events = []
    higgsRecoScores = []
    topRecoScores = []

    if channel=='3l':
        events3lF = []
        eventsDecay = []
        higgsRecoScoresF = []

    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
           
        #Get the events
        nom.GetEntry(idx)

        #Find the best top combination, top reco score
        topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
        if not topRes:
            topIdx0, topIdx1 = 0,0
            topScore = -10
        else:
            topIdx0, topIdx1 = topRes['bestComb']
            topScore = topRes['topScore']

        topRecoScores.append(topScore) # add the top reco score

        #Find the higgs decay products, higgs reco score for 3lF model
        if channel=='3l':
            res3lF = findBestHiggsTop(nom, '3lF', model3lF, normFactors3lF, topIdx0, topIdx1, topScore)
            #if not res3lF: continue
            higgsTopScoreF = res3lF['higgsTopScore']
            lepIdx = res3lF['bestComb'][0]

            events3lF.append( ptDictHiggsTop3lF(nom, lepIdx, higgsTopScoreF, topIdx0, topIdx1, topScore) )
            higgsRecoScoresF.append(higgsTopScoreF)

        #Find the higgs decay products, higgs reco score for 3lS, 2lSS (same final state)
        if channel=='3l':
            res = findBestHiggsTop(nom, '3lS', model3lS, normFactors3lS, topIdx0, topIdx1, topScore)
        else:
            res = findBestHiggsTop(nom, '2lSS', model2lSS, normFactors2lSS, topIdx0, topIdx1, topScore)
            
        if not res: 
            higgsTopScore = -10
            lepIdx, jetIdx0, jetIdx1 = 0,0,0
        else:
            higgsTopScore = res['higgsTopScore']
            lepIdx, jetIdx0, jetIdx1 = res['bestComb']
        
        #add the pt prediction dictionary
        events.append( ptDict(nom, jetIdx0, jetIdx1, lepIdx, higgsTopScore, topIdx0, topIdx1, topScore) )
        higgsRecoScores.append(higgsTopScore)

        #add decay mode dicts
        if channel=='3l':
            eventsDecay.append( decayDict(nom, higgsTopScoreF, higgsTopScore, topIdx0, topIdx1, topScore) )

    if channel=='3l':
        return events, events3lF, eventsDecay, higgsRecoScores, higgsRecoScoresF, topRecoScores
    else:
        return events, higgsRecoScores, topRecoScores

def makePrediction(events, kerasModel, normFactors):
    inDF = pd.DataFrame.from_dict(events)
    inDF = (inDF - normFactors[1])/(normFactors[0]-normFactors[1])
    yPred = kerasModel.predict(inDF.values)

    if len(normFactors)>2:
        yPred = yPred*normFactors[2]
    
    return yPred

def addToFile(outPath, y, name):
    with root_open(outPath, mode='a') as myfile:
        y = np.asarray(y)
        y.dtype = [(name, 'float32')]
        y.dtype.names = [name]
        root_numpy.array2tree(y, tree=myfile.nominal)
        myfile.write()
        myfile.Close()

def addPred(inf):
    if '3l' in inf:                                                                     
        channel='3l'

        ptModel = load_model("/data_ceph/afwebb/higgs_diff/ptPrediction/models/keras_model_higgsTop3lS.h5")
        ptNormFactors = np.load("/data_ceph/afwebb/higgs_diff/ptPrediction/models/higgsTop3lS_normFactors.npy", 
                                allow_pickle=True)

        binModel = load_model("/data_ceph/afwebb/higgs_diff/ptPrediction/models/bin_keras_model_higgsTop3lS.h5")
        binNormFactors = np.load("/data_ceph/afwebb/higgs_diff/ptPrediction/models/bin_higgsTop3lS_normFactors.npy", 
                                 allow_pickle=True)

        ptModelF = load_model("/data_ceph/afwebb/higgs_diff/ptPrediction/models/keras_model_higgsTop3lF.h5")
        ptNormFactorsF = np.load("/data_ceph/afwebb/higgs_diff/ptPrediction/models/higgsTop3lF_normFactors.npy", 
                                 allow_pickle=True)

        binModelF = load_model("/data_ceph/afwebb/higgs_diff/ptPrediction/models/bin_keras_model_higgsTop3lF.h5")      
        binNormFactorsF = np.load("/data_ceph/afwebb/higgs_diff/ptPrediction/models/bin_higgsTop3lF_normFactors.npy", 
                                  allow_pickle=True)

        decayModel = load_model("/data_ceph/afwebb/higgs_diff/decay3l/models/keras_model_decay3l.h5")       
        decayNormFactors = np.load("/data_ceph/afwebb/higgs_diff/decay3l/models/decay3l_normFactors.npy",
                                   allow_pickle=True)

        outReco = runReco(inf)
        if not outReco:
            return
        events, events3lF, eventsDecay, higgsRecoScores, higgsRecoScoresF, topRecoScores = outReco

    else:
        channel = '2lSS'

        ptModel = load_model("/data_ceph/afwebb/higgs_diff/ptPrediction/models/keras_model_higgsTop2lSS.h5")
        ptNormFactors = np.load("/data_ceph/afwebb/higgs_diff/ptPrediction/models/higgsTop2lSS_normFactors.npy", 
                                allow_pickle=True)

        binModel = load_model("/data_ceph/afwebb/higgs_diff/ptPrediction/models/bin_keras_model_higgsTop2lSS.h5")      
        binNormFactors = np.load("/data_ceph/afwebb/higgs_diff/ptPrediction/models/bin_higgsTop2lSS_normFactors.npy", 
                                 allow_pickle=True)

        outReco = runReco(inf)
        if not outReco:
            return
        events, higgsRecoScores, topRecoScores = outReco

    #run the events through the pt prediction model
    pt_pred = makePrediction(events, ptModel, ptNormFactors)
    bin_pred = makePrediction(events, binModel, binNormFactors)

    #add the prediction to the root file
    if channel=='3l':
        #do the same for 3l specific models - 3lF, decay
        pt_predF = makePrediction(events3lF, ptModelF, ptNormFactorsF)
        bin_predF = makePrediction(events3lF, binModelF, binNormFactorsF)
        decay_pred = makePrediction(eventsDecay, decayModel, decayNormFactors)
    
        addToFile(inf, pt_pred, 'recoHiggsPt_3lS')
        addToFile(inf, bin_pred, 'binHiggsPt_3lS')
        
        addToFile(inf, pt_predF, 'recoHiggsPt_3lF')                                                           
        addToFile(inf, bin_predF, 'binHiggsPt_3lF')
    
        addToFile(inf, decay_pred, 'decayScore')

    else:
        addToFile(inf, pt_pred, 'recoHiggsPt_2lSS')                                               
        addToFile(inf, bin_pred, 'binHiggsPt_2lSS')

linelist = [line.rstrip() for line in open(sys.argv[1])]
addPred(linelist[0])
#Parallel(n_jobs=12)(delayed(addPred)(inf) for inf in linelist)
