#Script to test whether the combination of leptons/jets with the highest xgb score corresponds to the correct combo
import ROOT
import pandas as pd
import numpy as np
#import uproot
import sys
import math
import pickle
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import xgboost as xgb
from dictTop2lSS import topDictFlat2lSS
import matplotlib.pyplot as plt
from functionsTop import selection2lSS, jetCombos2lSS

inputFile = sys.argv[1]
modelPath = sys.argv[2]
f=ROOT.TFile.Open(inputFile)
nom=f.Get('nominal')

totalPast = 0
nCorrect = 0
lepCorrect = 0
lep1jCorrect = 0
j1Correct = 0

events = []
bestScores = []

wrongTruth = []
wrongPred = []
rightTruth = []

truthScores = []
rightScores = []
wrongScores = []

xgbModel = pickle.load(open(modelPath, "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi


#la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*', 'trilep_type', 
#                            'nJets', 'nJets_MV2c10_70', 'jet_numTrk'])

def drCheck(eta, phi, truth_eta, truth_phi, cut):
    dr = sqrt( (phi-truth_phi)**2 + (eta-truth_eta)**2 )
  
    return dr < cut

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
nMatch = 0
higgVecs = []


fourVecDicts = []

#for idx in range(len(la[b'met']) ):
#    #current+=1
#    if idx%10000==0:
#        print(idx)                                                                                                 
#    if idx==50000:
#        break
nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==5000:       
        break                                                                                                                
    nom.GetEntry(idx)

    truthComb = []
    match=0

    truthBs = []
    for i in range(len(nom.jet_pt)):                                                                                   
        if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthflav[i])==5:                                           
            truthComb.append(i)

    if len(truthComb)!=2: continue
    
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

    topDF = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(topDF, feature_names=list(topDF))

    #print([x[1] for x in combos])

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    jetMatches = bestComb

    for c, s in zip(combos, pred):
        if c[1][0] in truthComb and c[1][1] in truthComb:
            truthScores.append(s)

    totalPast+=1

    if sorted(bestComb)==sorted(truthComb):
        nCorrect+=1
        rightScores.append(pred[best])
        #rightTruth.append(truthK)
    else:
        wrongScores.append(pred[best])
        #wrongTruth.append(truthK)
        #wrongPred.append(predK)

    if bestComb[0] in truthComb or bestComb[1] in truthComb:
        j1Correct+=1

print('events passed', totalPast)
print('percent correct', nCorrect/totalPast)
#print('lep correct', lepCorrect/totalPast)
print('1 jet correct', j1Correct/totalPast)
#print('lep and 1 jet correct', lep1jCorrect/totalPast)
'''
rightTruthDF = pd.DataFrame(rightTruth)
wrongTruthDF = pd.DataFrame(wrongTruth)
wrongPredDF = pd.DataFrame(wrongPred)

rightTruthDF.to_csv('outputData/rightTruthAllBad.csv', index=False)
wrongTruthDF.to_csv('outputData/wrongTruthAllBad.csv', index=False)
wrongPredDF.to_csv('outputData/wrongPredAllBad.csv', index=False)


plt.figure()
plt.hist(truthScores, 30)
plt.savefig('plots/topTruthScores.png')

plt.figure()
plt.hist(rightScores, 30, alpha=0.5, label='right')
plt.hist(wrongScores, 30, alpha=0.5, label='wrong')
plt.legend()
plt.savefig('plots/topMatchScores.png')


'''
