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
from dictHiggs import higgsTopDict2lSS, higgsTopDict3lS, higgsTopDict3lF
from functionsMatch import jetCombosTop, findBestTopKeras
from joblib import Parallel, delayed
import multiprocessing

#Open input file
#inf = sys.argv[1]
#outDir = sys.argv[2]

def runReco(inf):
    #Set the channel, load in the top model
    if '3l' in inf:
        channel='3l'
        topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/keras_model_top3l.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/top3l_normFactors.npy")
        flatDict = higgsTopDict3lS
        is3l = True
    elif '2lSS' in inf:
        channel='2lSS'
        topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/keras_model_top2lSS.h5")
        topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/top2lSS_normFactors.npy")
        flatDict = higgsTopDict2lSS
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
        topIdx0, topIdx1 = topRes['bestComb']
        topScore = topRes['topScore']

        #identify which lepton came from the Higgs
        lepIdx = -1
        if nom.lep_Parent_0 == 25: 
            if is3l: 
                isF = True
            else: 
                lepIdx = 0 #lep0 is always from the Higgs in 3l case
                isF = False
        else:
            isF = False
        if nom.lep_Parent_1 == 25: lepIdx = 1
        if is3l and nom.lep_Parent_2 == 25: lepIdx = 2

        if lepIdx == -1: continue

        #Get index of the non-higgs decay lepton
        if is3l: wrongLep = (lepIdx)%2+1
        else: wrongLep = (lepIdx+1)%2

        if isF:
            events3lF.append( higgsTopDict3lF(nom, lepIdx, topIdx0, topIdx1, topScore, lepIdx-1) ) #Correct combination
            #events3lF.append( higgsTopDict3lF(nom, lepIdx, topIdx0, topIdx1, topScore, 1) )
            #events3lF.append( higgsTopDict3lF(nom, wrongLep, topIdx0, topIdx1, topScore, 0) ) #Incorrect combination - swaps 2 and 1
        else:
            higgsJets = []
            badJets = []

            #Get indices of jets from Higgs
            for i in range(len(nom.jet_pt)):
                if i == topIdx0 or i == topIdx1: continue
                if nom.jet_jvt[i]<0.59: continue
                if abs(nom.jet_parents[i])==25:
                    higgsJets.append(i)
                else:
                    badJets.append(i)
        
            if len(higgsJets)!=2: continue #Only include events where both jets are truth matched to the Higgs
            
            events.append( flatDict( nom, higgsJets[0], higgsJets[1], lepIdx, topIdx0, topIdx1, topScore, 1 ) ) #Correct combination
            events.append( flatDict( nom, higgsJets[0], higgsJets[1], wrongLep, topIdx0, topIdx1, topScore, 0 ) ) #Wrong lepton, right jets
            
            #for l in range(min([1, len(badJets)]) ):
            if len(badJets)>1:
                #right lepton, one correct jet
                events.append( flatDict( nom, higgsJets[0], random.sample(badJets,1)[0], lepIdx, topIdx0, topIdx1, topScore, 0 ) ) 
                events.append( flatDict( nom, random.sample(badJets,1)[0], higgsJets[1], lepIdx, topIdx0, topIdx1, topScore, 0 ) )
                
                #wrong lepton, one correct jet
                events.append( flatDict( nom, higgsJets[0], random.sample(badJets,1)[0], wrongLep, topIdx0, topIdx1, topScore, 0 ) ) 
                events.append( flatDict( nom, random.sample(badJets,1)[0], higgsJets[1], wrongLep, topIdx0, topIdx1, topScore, 0 ) )

            #Right lepton, wrong jets
            for l in range(min([2, len(badJets)]) ):
                if len(badJets)>2:
                    i,j = random.sample(badJets,2)
                    events.append( flatDict( nom, i, j, lepIdx, topIdx0, topIdx1, topScore, 0 ) )

            #Wrong leptons, wrong jets
            for l in range(min([4, len(badJets)]) ):
                if len(badJets)>2:
                    i,j = random.sample(badJets,2)
                    events.append( flatDict( nom, i, j, wrongLep, topIdx0, topIdx1, topScore, 0 ) )

    dfFlat = pd.DataFrame.from_dict(events)
    dfFlat = shuffle(dfFlat)

    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='2lSS':
        dfFlat.to_csv('csvFiles/higgsTop2lSS/'+outF, index=False)
    elif channel=='3l':
        dfFlat.to_csv('csvFiles/higgsTop3lS/'+outF, index=False)
        df3lF = pd.DataFrame.from_dict(events3lF)
        df3lF = shuffle(df3lF)
        df3lF.to_csv('csvFiles/higgsTop3lF/'+outF, index=False)

linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=15)(delayed(runReco)(inf) for inf in linelist)
