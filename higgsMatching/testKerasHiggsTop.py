''' 
Test the accuracey of the keras top matching algorithm for 2lSS  
Outputs the percentage of events the model correctly identifies both b-jets, and at least one b-jet
Usage: python3.6 testKerasHiggs.py <inputFile> <kerasModel> <kerasNormFactors>
'''

#import keras
from tensorflow.keras.models import load_model
import ROOT
import pandas as pd
import numpy as np
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
import pickle
from functionsMatch import selection2lSS, higgsTopCombos, findBestTopKeras, findBestHiggsTop

#Load in the input file
inf = sys.argv[1]
f = ROOT.TFile.Open(inf, "READ")
nom = f.Get('nominal')

#Load in the top matching keras model
higgsModel = load_model(sys.argv[2])
higgsModel.compile(loss="binary_crossentropy", optimizer='adam')

if '2lSS' in sys.argv[2]:
    channel = '2lSS'
    topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top2lSS.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top2lSS_normFactors.npy")
elif '3lF' in sys.argv[2]:
    channel = '3lF'
    topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy")
elif '3lS' in sys.argv[2]:
    channel = '3lS'
    topModel = load_model("/data_ceph/afwebb/higgs_diff/topMatching/models/keras_model_top3l.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/topMatching/models/top3l_normFactors.npy")
else:
    print('Cannot determine which channel to use')
    exit

higgsNormFactors = np.load(sys.argv[3])
higgsMaxVals = higgsNormFactors[0]
higgsMinVals = higgsNormFactors[1]                                                                                         
higgsDiff = higgsMaxVals - higgsMinVals

events = []
nEntries = nom.GetEntries()

nEvents=0
nCorrect=0
lepCorrect=0
oneCorrect=0

#Loop over each entry, add to events dict
for idx in range(nEntries):
    if idx%1000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==5000:
        break

    nom.GetEntry(idx)

    if '3l' in channel:
        topRes = findBestTopKeras(nom, '3l', topModel, topNormFactors)
    else:
        topRes = findBestTopKeras(nom, '2lSS', topModel, topNormFactors)

    topIdx0, topIdx1 = topRes['bestComb']
    topScore = topRes['topScore']
    print(topScore)
    #Get dict of all possible jet combinations
    higgsRes = findBestHiggsTop(nom, channel, higgsModel, higgsNormFactors, topIdx0, topIdx1, topScore)
    higgsMatches = higgsRes['bestComb']
    truthPair = higgsRes['truthComb']
    if channel == '3lF' and len(truthPair)!=1:                                                                                 
        continue                                                                                                               
    elif channel!='3lF' and len(truthPair)!=3:                                                                                 
        continue 

    print(higgsRes['higgsTopScore'])

    nEvents+=1
    if higgsMatches == truthPair:
        nCorrect+=1
    if higgsMatches[0] == truthPair[0]:
        lepCorrect+=1
        if channel != '3lF':
            if higgsMatches[1] in truthPair[1:] or higgsMatches[2] in truthPair[1:]:
                oneCorrect+=1

outRes = open(f'models/testCorrect{channel}.txt', 'a')
outRes.write('\n\n')
outRes.write(f"{channel} Higgs Top Matching Tested on {inf}\n")
if channel != '3lF':
    outRes.write(f"Correct: {str(round(nCorrect/nEvents, 2))}\n")
    outRes.write(f"Lepton Correct: {str(round(lepCorrect/nEvents, 2))}\n")
    outRes.write(f"Lepton, One Jet Correct: {str(round(oneCorrect/nEvents, 2))}\n\n")
else:
    outRes.write(f"Correct: {str(round(nCorrect/nEvents, 2))}\n\n")
    
