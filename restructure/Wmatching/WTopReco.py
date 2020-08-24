'''
Inputs a ROOT file, outputs a csv file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles(2lSS 3l)/{flat, fourVec}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 higgsTopReco.py <input root file> <channel>
'''

import pandas as pd
from tensorflow.keras.models import load_model
import ROOT
from ROOT import TFile
from sklearn.utils import shuffle
import numpy as np
import sys
import random
from dictW import WTopDict2lSS, WTopDict3l
from functionsW import jetCombosTop, findBestTopKeras
from joblib import Parallel, delayed
import multiprocessing

#Open input file
#inf = sys.argv[1]
#outDir = sys.argv[2]

def runReco(inf):
    #Set the channel, load in the top model
    if '3l' in inf:
        channel='3l'
        topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/keras_model_top3l.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/top3l_normFactors.npy")
        is3l = True
    elif '2lSS' in inf:
        channel='2lSS'
        topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/keras_model_top2lSS.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/top2lSS_normFactors.npy")
        is3l = False
    else:
        print(f'Channel {channel} is invalid. Should be 2lSS or 3l')
        exit()
        
    print('loaded')
    topMaxVals = topNormFactors[0]                                                                                      
    topMinVals = topNormFactors[1]
    topDiff = topMaxVals - topMinVals
    
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    
    #initialize output dicts
    events = []
    events3lF = []
    
    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
            
        nom.GetEntry(idx)

        topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
        if not topRes:
            continue
        topIdx0, topIdx1 = topRes['bestComb']
        topScore = topRes['topScore']

        #identify which lepton came from the Higgs
        lepIdx = -1
        if abs(nom.lep_Parent_0) == 24: 
            if not is3l: lepIdx = 0 #lep0 is always from the Higgs in 3l case
        if abs(nom.lep_Parent_1) == 24: lepIdx = 1
        if is3l and abs(nom.lep_Parent_2) == 24: lepIdx = 2

        if lepIdx == -1: continue #Exclude events where none of the leptons are from the W

        #Get index of the non-higgs decay lepton
        if is3l: wrongLep = (lepIdx)%2+1
        else: wrongLep = (lepIdx+1)%2

        if is3l:
            events.append( WTopDict3l(nom, lepIdx, topIdx0, topIdx1, topScore, 1) ) #Correct combination
            events.append( WTopDict3l(nom, wrongLep, topIdx0, topIdx1, topScore, 0) ) #Incorrect combination - swaps 2 and 1
        else:                                                                                                  
            events.append( WTopDict2lSS(nom, lepIdx, topIdx0, topIdx1, topScore, 1) ) #Correct combination        
            events.append( WTopDict2lSS(nom, wrongLep, topIdx0, topIdx1, topScore, 0) ) #Incorrect combination - swaps 2 and\ 1  

    dfFlat = pd.DataFrame.from_dict(events)
    dfFlat = shuffle(dfFlat)

    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='2lSS':
        dfFlat.to_csv('csvFiles/WTop2lSS/'+outF, index=False)
    elif channel=='3l':
        dfFlat.to_csv('csvFiles/WTop3l/'+outF, index=False)

linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=8)(delayed(runReco)(inf) for inf in linelist)
