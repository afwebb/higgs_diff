''' 
Test the accuracey of the keras top matching algorithm for 2lSS  
Outputs the percentage of events the model correctly identifies both b-jets, and at least one b-jet
Usage: python3.6 testKerasTop2lSS.py <inputFile> <kerasModel> <kerasNormFactors>
'''

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
import keras
from keras.models import load_model
import pickle
from functionsTop import selection2lSS, jetCombos2lSS, jetCombos3l

#Load in the input file
inf = sys.argv[1]
f = ROOT.TFile.Open(inf, "READ")
nom = f.Get('nominal')

#Load in the top matching keras model
topModel = load_model(sys.argv[2])
topModel.compile(loss="binary_crossentropy", optimizer='adam')

topNormFactors = np.load(sys.argv[3])
topMaxVals = topNormFactors[0]
topMinVals = topNormFactors[1]                                                                                            
topDiff = topMaxVals - topMinVals

events = []
nEntries = nom.GetEntries()

nEvents=0
nCorrect=0
oneCorrect=0

#Loop over each entry, add to events dict
for idx in range(nEntries):
    if idx%1000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==5000:
        break

    nom.GetEntry(idx)

    #Apply 2lSS preselection                                                                                               
    #if not selection2lSS(nom): 
    #    continue

    #truthBs = []
    #for i in range(len(nom.jet_pt)):
    #    if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthflav[i])==5:
    #        truthBs.append(i)

    #if len(truthBs)!=2: continue

    #Get dict of all possible jet combinations
    if '2lSS' in inf:
        combosTop = jetCombos2lSS(nom, 0)
    elif '3l' in inf:
        combosTop = jetCombos3l(nom, 0)
    else:
        'not sure which channel to use'
        break

    truthBs = combosTop['truthComb']
    if len(truthBs)!=2: continue

    if 'flat' in sys.argv[2]:
        topDF = pd.DataFrame.from_dict(combosTop['flatDicts'])
    else:
        topDF = pd.DataFrame.from_dict(combosTop['fourVecDicts'])

    #find combination of jets with highest top score
    topDF=(topDF - topMinVals)/(topDiff)                                                                           
    topMat = topDF.values

    topPred = topModel.predict(topMat)
    topBest = np.argmax(topPred)

    topMatches = combosTop['jetIdx'][topBest]

    nEvents+=1
    #if abs(nom.jet_parents[topMatches[0]])==6 and abs(nom.jet_parents[topMatches[1]])==6:
    if topMatches[0] in truthBs and topMatches[1] in truthBs:
        nCorrect+=1
    #if abs(nom.jet_parents[topMatches[0]])==6 or abs(nom.jet_parents[topMatches[1]])==6:
    if topMatches[0] in truthBs or topMatches[1] in truthBs:
        oneCorrect+=1


print("Both Correct", nCorrect/nEvents)
print("One Correct", oneCorrect/nEvents)
