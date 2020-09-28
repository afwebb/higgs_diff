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
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
import pickle
#from functionsMatch import selection2lSS, jetCombosTop2lSS, jetCombosTop3l, findBestTopKeras
from functionsMatch import jetCombosTop, findBestTopKeras

#Load in the input file
inf = sys.argv[1]
f = ROOT.TFile.Open(inf, "READ")
nom = f.Get('nominal')

if '2lSS' in inf:
    channel = '2lSS'
else:
    channel = '3l'

#Load in the top matching keras model
topModel = load_model(sys.argv[2])
topModel.compile(loss="binary_crossentropy", optimizer='adam')

topNormFactors = np.load(sys.argv[3])
topMaxVals = topNormFactors[0]
topMinVals = topNormFactors[1]                                                                                            
topDiff = topMaxVals - topMinVals

events = []
badEvents = pd.DataFrame()
nEntries = nom.GetEntries()

nEvents, nCorrect,oneCorrect=0,0,0
nGood,nGoodCorrect,nGoodOne=0,0,0
nBad,nBadCorrect,nBadOne=0,0,0
n1b, n2b, n3b = 0,0,0
n1bCorrect, n2bCorrect, n3bCorrect = 0,0,0

#Loop over each entry, add to events dict
for idx in range(nEntries):
    if idx%1000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==5000:
        break

    nom.GetEntry(idx)

    topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
    topMatches, truthBs, topScore = topRes['bestComb'], topRes['truthComb'], topRes['topScore']
    #print(topRes['bestComb'], topRes['truthComb'], topRes['topScore'])
    #topMatches, truthBs = findBestTopKeras(nom, channel, topModel, topNormFactors, 1)

    if len(truthBs)!=2: continue 

    if topScore>0.3:
        nGood+=1
        if topMatches[0] in truthBs and topMatches[1] in truthBs:
            nGoodCorrect+=1
        if topMatches[0] in truthBs or topMatches[1] in truthBs:
            nGoodOne+=1
    if topScore<0.3:
        nBad+=1
        if topMatches[0] in truthBs and topMatches[1] in truthBs:
            nBadCorrect+=1                                                                                                  
        if topMatches[0] in truthBs or topMatches[1] in truthBs:                                                            
            nBadOne+=1

    nEvents+=1

    if nom.nJets_OR_DL1r_70==1:
        n1b+=1
    elif nom.nJets_OR_DL1r_70==2:
        n2b+=1
    else:
        n3b+=1

    #if abs(nom.jet_parents[topMatches[0]])==6 and abs(nom.jet_parents[topMatches[1]])==6:
    if topMatches[0] in truthBs and topMatches[1] in truthBs:
        nCorrect+=1
        if nom.nJets_OR_DL1r_70==1:                                                                                        
            n1bCorrect+=1                                                                                            
        elif nom.nJets_OR_DL1r_70==2:                                                                                
            n2bCorrect+=1                                                                                                  
        else:
            n3bCorrect+=1
    else:
        #print(badEvents.transpose())
        if badEvents.empty:                                                                                              
            badEvents = topRes['truthDF']
            badEvents = pd.concat([badEvents, topRes['bestDF']], axis=1)  
        else:                                                                                                           
            badEvents = pd.concat([badEvents,topRes['truthDF'],topRes['bestDF']], axis=1)
    #if abs(nom.jet_parents[topMatches[0]])==6 or abs(nom.jet_parents[topMatches[1]])==6:
    if topMatches[0] in truthBs or topMatches[1] in truthBs:
        oneCorrect+=1

#dfTop = pd.DataFrame.from_dict(badEvents.transpose())
badEvents.transpose().to_csv('csvFiles/top2lSS/badAllTop', index=False)

outRes = open(f'models/testCorrect{channel}.txt', 'a')
outRes.write(f"{channel} Top Matching Tested on {inf}, {nEvents} events\n")
outRes.write(f"model used: {sys.argv[2]}\n")
outRes.write(f"Correct: {str(round(nCorrect/nEvents, 3))}\n")                                                                
outRes.write(f"One Jet Correct: {str(round(oneCorrect/nEvents, 3))}\n\n")

outRes.write(f"{channel} Top Score > 0.6, {nGood} events\n")
outRes.write(f"Correct: {str(round(nGoodCorrect/nGood, 3))}\n")
outRes.write(f"One Jet Correct: {str(round(nGoodOne/nGood, 3))}\n\n")

outRes.write(f"{channel} Top Score < 0.6, {nBad} events\n")
outRes.write(f"Correct: {str(round(nBadCorrect/nBad, 3))}\n")                                                              
outRes.write(f"One Jet Correct: {str(round(nBadOne/nBad, 3))}\n\n\n")

outRes.write(f"1b Correct: {str(round(n1bCorrect/n1b, 3))}\n")
outRes.write(f"2b Correct: {str(round(n2bCorrect/n2b, 3))}\n")
outRes.write(f">=3b Correct: {str(round(n3bCorrect/n3b, 3))}\n")
