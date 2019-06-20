#Script to test whether the combination of leptons/jets with the highest xgb score corresponds to the correct combo

import ROOT
import pandas as pd
import numpy as np
import uproot
import sys
import math
import pickle
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import xgboost as xgb
from dict_higgs2l import higgsDict
import matplotlib.pyplot as plt

inputFile = sys.argv[1]
modelPath = sys.argv[2]
f=uproot.open(inputFile)
nom=f.get('nominal')

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


la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*', 'trilep_type', 
                            'nJets', 'nJets_MV2c10_70', 'jet_numTrk'])

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

for idx in range(len(la[b'met']) ):
    #current+=1
    if idx%10000==0:
        print(idx)                                                                                                 
    if idx==50000:
        break

    if la[b'trilep_type'][idx]==0: continue
    if la[b'nJets'][idx]<2: continue
    if la[b'nJets_MV2c10_70'][idx]==0: continue
    if len(la[b'lep_pt'][idx])!=3: continue

    truthComb = []

    higgCand = LorentzVector()
    
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])
    
    fourVecs['met'] = met
    
    higgCand+=met
        
    lep4Vecs = []

    #print(la[b'lep_parent'][idx])

    lepMatch = []
    badLep = -1
    for i in range(3):

        lep_pt = la[b'lep_pt'][idx][i]
        lep_eta = la[b'lep_eta'][idx][i]
        lep_phi = la[b'lep_phi'][idx][i]
        lep_E = la[b'lep_E'][idx][i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        lep4Vecs.append(lepVec)

        if la[b'lep_parent'][idx][i]==25:
            lepMatch.append(i)
        else:
            badLep = i

    if len(lepMatch)==1:
        decay = "semiLep"
    elif len(lepMatch)==2 or len(lepMatch)==0:
        decay = "fullLep"
    else:
        #print(len(lepMatch))
        #print(la[b'lep_parent'][idx])
        continue

    combos = []

    possCombs = [[0,1,2],[0,2,1]]
    for comb in possCombs:
        k = higgsDict( lep4Vecs[ comb[0] ], lep4Vecs[ comb[1] ], lep4Vecs[ comb[2] ], met)
        combos.append([k, [comb[0], comb[1]] ])

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    lepMatch = bestComb[0]
    jetMatches = bestComb[1:]
    '''
    for c, s in zip(combos, pred):
        if c[1][0]==truthComb[0] and c[1][1] in truthComb[1] and c[1][2] in truthComb[1:]:
            #truthK = {**c[0], **c[1]}
            truthScores.append(s)
            #print('truth score', s)
    '''
    #predK = {**combos[best][0], **combos[best][1]}

    #print(bestComb, truthComb, pred[best])

    totalPast+=1

    if decay == "semiLep":
        rightScores.append(pred[best])
    elif decay == "fullLep":
        wrongScores.append(pred[best])

print('events passed', totalPast)
print('percent correct', nCorrect/totalPast)
#print('lep correct', lepCorrect/totalPast)
#print('1 jet correct', j1Correct/totalPast)
#print('lep and 1 jet correct', lep1jCorrect/totalPast)
'''
rightTruthDF = pd.DataFrame(rightTruth)
wrongTruthDF = pd.DataFrame(wrongTruth)
wrongPredDF = pd.DataFrame(wrongPred)

rightTruthDF.to_csv('outputData/rightTruthAllBad.csv', index=False)
wrongTruthDF.to_csv('outputData/wrongTruthAllBad.csv', index=False)
wrongPredDF.to_csv('outputData/wrongPredAllBad.csv', index=False)

'''
#plt.figure()
#plt.hist(truthScores, 30)
#plt.savefig('plots/higgs2lTruthScores.png')

plt.figure()
plt.hist(rightScores, 30, alpha=0.5, label='semilep')
plt.hist(wrongScores[:len(rightScores)], 30, alpha=0.5, label='full lep')
plt.legend()
plt.savefig('plots/bad2lMatchScores.png')


