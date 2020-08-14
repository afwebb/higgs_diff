'''
Inputs a ROOT file, outputs a csv file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles(2lSS 3l)/{flat, fourVec}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 higgsTopReco.py <input root file> <channel>
'''

import ROOT
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
import keras
from keras.models import load_model
from dictHiggs import higgsTopDict2lSS, higgsTopDict3lS, higgsTopDict3lF
import functionsTop
from functionsTop import jetCombos2lSS, jetCombos3l

#Open input file
inf = sys.argv[1]
outDir = sys.argv[2]

#Set the channel, load in the top model
if outDir=='3l':
    topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/keras_model_flat3l.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/flat3l_normFactors.npy")
    flatDict = higgsTopDict3lS
    is3l = True
elif outDir=='2lSS':
    topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/keras_model_flat3l.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/topMatching/models/flat3l_normFactors.npy")
    flatDict = higgsTopDict2lSS
    is3l = False
else:
    print(f'Channel {outDir} is invalid. Should be 2lSS or 3l')
    exit()

topMaxVals = topNormFactors[0]                                                                                               
topMinVals = topNormFactors[1]
topDiff = topMaxVals - topMinVals

f = ROOT.TFile.Open(inf)
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
    
    #Get all possible combinations of tops
    if outDir=='2lSS':
        combosTop = jetCombos2lSS(nom, 0)
    elif outDir=='3l':
        combosTop = jetCombos3l(nom, 0)
    else:
        'not sure which channel to use'
        break

    topDF = pd.DataFrame.from_dict(combosTop['flatDicts']) #Convert dict to dataframe
    topDF=(topDF - topMinVals)/(topDiff) #Normalize dataframe
    topPred = topModel.predict(topDF.values) #Feed df into NN, predict which pair is best
    topIdx0, topIdx1 = combosTop['jetIdx'][np.argmax(topPred)] #Get the indices of the pair with the highest score

    #identify which lepton came from the Higgs
    lepIdx = -1
    if nom.lep_Parent_0 == 25: 
        isF = True
        if not is3l: lepIdx = 0 #lep0 is always from the Higgs in 3l case
    else:
        isF = False
    if nom.lep_Parent_1 == 25: lepIdx = 1
    if is3l and nom.lep_Parent_2 == 25: lepIdx = 2

    if lepIdx == -1: continue

    #Get index of the non-higgs decay lepton
    if is3l:
        wrongLep = (lepIdx)%2+1
    else:
        wrongLep = (lepIdx+1)%2

    if isF:
        events3lF.append( higgsTopDict3lF(nom, lepIdx, topIdx0, topIdx1, 1) ) #Correct combination
        events3lF.append( higgsTopDict3lF(nom, wrongLep, topIdx0, topIdx1, 0) ) #Incorrect combination - swaps 2 and 1
    else:
        higgsJets = []
        badJets = []

        #Get indices of jets from Higgs
        for i in range(len(nom.jet_pt)):
            if i == topIdx1 or i == topIdx2: continue
            if nom.jet_jvt[i]<0.59: continue
            if abs(nom.jet_parents[i])==25:
                higgsJets.append(i)
            else:
                badJets.append(i)
        
        if len(higgsJets)!=2: continue #Only include events where both jets are truth matched to the Higgs

        events.append( flatDict( nom, higgsJets[0], higgsJets[1], lepIdx, topIdx0, topIdx1, 1 ) ) #Correct combination

        events.append( flatDict( nom, higgsJets[0], higgsJets[1], wrongLep, topIdx0, topIdx1, 0 ) ) #Wrong lepton, right jets
        
        #for l in range(min([1, len(badJets)]) ):
        if len(badJets)>1:
            #right lepton, one correct jet
            events.append( flatDict( nom, higgsJets[0], random.sample(badJets,1)[0], lepIdx, topIdx0, topIdx1, 0 ) ) 
            events.append( flatDict( nom, random.sample(badJets,1)[0], higgsJets[1], lepIdx, topIdx0, topIdx1, 0 ) )
            
            #wrong lepton, one correct jet
            events.append( flatDict( nom, higgsJets[0], random.sample(badJets,1)[0], wrongLep, topIdx0, topIdx1, 0 ) ) 
            events.append( flatDict( nom, random.sample(badJets,1)[0], higgsJets[1], wrongLep, topIdx0, topIdx1, 0 ) )

        #Right lepton, wrong jets
        for l in range(min([2, len(badJets)]) ):
            if len(badJets)>2:
                i,j = random.sample(badJets,2)
                events.append( flatDict( nom, i, j, lepIdx, topIdx0, topIdx1, 0 ) )

        #Wrong leptons, wrong jets
        for l in range(min([4, len(badJets)]) ):
            if len(badJets)>2:
                i,j = random.sample(badJets,2)
                events.append( flatDict( nom, i, j, wrongLep, topIdx0, topIdx1, 0 ) )

dfFlat = pd.DataFrame.from_dict(events)
dfFlat = shuffle(dfFlat)

outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
if outDir=='2lSS':
    dfFlat.to_csv('csvFilesTop2lSS/'+outF, index=False)
elif outDir=='3l':
    dfFlat.to_csv('csvFilesTop3lS/'+outF, index=False)

    df3lF = pd.DataFrame.from_dict(events3lF)
    df3lF = shuffle(df3lF)
    df3lF.to_csv('csvFilesTop3lF/'+outF, index=False)
