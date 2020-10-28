'''
Inputs a ROOT file, outputs a csv file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles(2lSS 3l)/{flat, fourVec}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 higgsTopReco.py <input root file> <channel>
'''

from tensorflow.keras.models import load_model
import ROOT
from ROOT import TFile
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import rootpy.io
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
from dictDecay import decayDict
from functionsMatch import findBestTopKeras, findBestHiggsTop
from joblib import Parallel, delayed
import multiprocessing

def runReco(inf):

    #load in the top model - not picklable, can't do outside the function 
    topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy")
    topMaxVals = topNormFactors[0]
    topMinVals = topNormFactors[1]
    topDiff = topMaxVals - topMinVals

    model3lF = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop3lF.h5")
    normFactors3lF = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop3lF_normFactors.npy")
    maxVals3lF, minVals3lF = normFactors3lF
    diff3lF = maxVals3lF - minVals3lF

    model3lS = load_model("/data_ceph/afwebb/higgs_diff/higgsMatching/models/keras_model_higgsTop3lS.h5")        
    normFactors3lS = np.load("/data_ceph/afwebb/higgs_diff/higgsMatching/models/higgsTop3lS_normFactors.npy")
    maxVals3lS, minVals3lS = normFactors3lS                                                                    
    diff3lS = maxVals3lS - minVals3lS

    f = TFile.Open(inf)
    nom = f.Get('nominal')

    #initialize output dicts
    events = []

    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
            
        nom.GetEntry(idx)

        #Perform top matching. Get top candidates, topScore
        topRes = findBestTopKeras(nom, '3l', topModel, topNormFactors)
        if not topRes: continue
        topIdx0, topIdx1 = topRes['bestComb']
        topScore = topRes['topScore']

        #Perform higgs matching. Get 3lF, 3lS scores
        res3lF = findBestHiggsTop(nom, '3lF', model3lF, normFactors3lF, topIdx0, topIdx1, topScore)
        res3lS = findBestHiggsTop(nom, '3lS', model3lS, normFactors3lS, topIdx0, topIdx1, topScore)

        if not res3lF or not res3lS: continue
        #identify which lepton came from the Higgs
        lepIdx = -1
        if nom.lep_Parent_0 == 25: 
            isF = True
        else:
            isF = False
        if nom.lep_Parent_1 == 25: lepIdx = 1
        if nom.lep_Parent_2 == 25: lepIdx = 2
            
        if lepIdx == -1: continue

        if isF:
            events.append( decayDict(nom, res3lF['higgsTopScore'], res3lS['higgsTopScore'], 
                                     topIdx0, topIdx1, topScore, 0) ) #Correct combination
        else:
            events.append( decayDict(nom, res3lF['higgsTopScore'], res3lS['higgsTopScore'],
                                     topIdx0, topIdx1, topScore, 1) ) #Incorrect combination - swaps 2 and 1

    dfFlat = pd.DataFrame.from_dict(events)
    dfFlat = shuffle(dfFlat)
    
    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    dfFlat.to_csv('csvFiles/'+outF, index=False)
    
linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=12)(delayed(runReco)(inf) for inf in linelist)
