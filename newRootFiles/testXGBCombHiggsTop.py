#Script to test whether the combination of leptons/jets with the highest xgb score corresponds to the correct combo

import ROOT
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
import pandas as pd
from dict_top import topDict
from dict_higgsTop import higgsTopDict
import matplotlib.pyplot as plt

inf = sys.argv[1]
modelPath = sys.argv[2]
topModelPath = sys.argv[3]
f=ROOT.TFile(inf, "READ")
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
badScores = []

xgbModel = pickle.load(open(modelPath, "rb"))
topModel = pickle.load(open(topModelPath, "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi


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

nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==2000:
        break

    nom.GetEntry(idx)

    truthComb = []

    higgCand = LorentzVector()
    
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met, 0, nom.met_phi, nom.met)
    
    fourVecs['met'] = met
    
    higgCand+=met
        
    lep4Vecs = []

    lepMatch = -1
    for i in range(2):

        lep_pt = nom.lep_pt[i]
        lep_eta = nom.lep_eta[i]
        lep_phi = nom.lep_phi[i]
        lep_E = nom.lep_E[i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        lep4Vecs.append(lepVec)

        if nom.lep_parent[i]==25:
            lepMatch=i

    if lepMatch == -1:
        continue

    truthComb.append(lepMatch)

    match=0  
    truthJets = []

    higgsJets=[]
    higgsJetsMV2c10=[]

    badJets=[]
    badJetsMV2c10=[]

    jet4Vecs = []
    jet4VecsMV2c10 = []

    match = 0

    for i in range(len(nom.jet_pt)):#nom.selected_jets'][i]:

        jet_pt = nom.jet_pt[i]
        jet_eta = nom.jet_eta[i]
        jet_phi = nom.jet_phi[i]
        jet_E = nom.jet_E[i]
        jet_MV2c10 = nom.jet_MV2c10[i]
        
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)
        
        jet4Vecs.append(jetVec)
        jet4VecsMV2c10.append(jet_MV2c10)
        
        if nom.jet_parent[i]==25:
            truthJets.append(i)
            higgCand+=jetVec
            higgsJets.append(jetVec)
            higgsJetsMV2c10.append(jet_MV2c10)
            match+=1
            truthComb.append(i)
        else:
            badJets.append(jetVec)
            badJetsMV2c10.append(jet_MV2c10)
               
    if match!=2: continue

    combosTop = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]

                t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j],
                             nom.jet_jvt[i], nom.jet_jvt[j],
                             nom.jet_numTrk[i], nom.jet_numTrk[j]
                         )

                combosTop.append([t, comb])

    #loop over combinations, score them in the BDT, figure out the best result                                                    
    topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
    topMat = xgb.DMatrix(topDF, feature_names=list(topDF))

    topPred = topModel.predict(topMat)
    topBest = np.argmax(topPred)

    bestTopComb = combosTop[topBest][1]
    topMatches = bestTopComb[1:]

    combos = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]

                if l==0:
                    k = higgsTopDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j],
                                      jet4Vecs[ topMatches[0] ], jet4Vecs[ topMatches[1] ], lep4Vecs[1],
                                      nom.jet_jvt[i], nom.jet_jvt[j],
                                      nom.jet_numTrk[i], nom.jet_numTrk[j]
                                  )
                else:
                    k = higgsTopDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[1], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j],
                                      jet4Vecs[ topMatches[0] ], jet4Vecs[ topMatches[1] ], lep4Vecs[0],
                                      nom.jet_jvt[i], nom.jet_jvt[j],
                                      nom.jet_numTrk[i], nom.jet_numTrk[j]
                                  )
                k['topScore'] = topPred[topBest]
                combos.append([k, comb])

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    lepMatch = bestComb[0]
    jetMatches = bestComb[1:]

    for c, s in zip(combos, pred):
        if c[1][0]==truthComb[0] and c[1][1] in truthComb[1:] and c[1][2] in truthComb[1:]:
            #truthK = {**c[0], **c[1]}
            #print('truth score', s)
            truthScores.append(s)
        else:
            badScores.append(s)
    #predK = {**combos[best][0], **combos[best][1]}

    #print(bestComb, truthComb, pred[best])

    totalPast+=1

    if sorted(bestComb)==sorted(truthComb):
        rightScores.append(pred[best])
    #    nCorrect+=1
        #rightTruth.append(truthK)
    else:
        wrongScores.append(pred[best])
    #    wrongTruth.append(truthK)
    #    wrongPred.append(predK)

    if bestComb[0]==truthComb[0] and (bestComb[1] in truthComb[1:] and bestComb[2] in truthComb[1:]):
        nCorrect+=1
    if bestComb[0]==truthComb[0]:
        lepCorrect+=1
    if bestComb[1] in truthComb[1:] or bestComb[2] in truthComb[1:]:
        j1Correct+=1
    if bestComb[0]==truthComb[0] and (bestComb[1] in truthComb[1:] or bestComb[2] in truthComb[1:]):
        lep1jCorrect+=1

print('events passed', totalPast)
print('percent correct', nCorrect/totalPast)
print('lep correct', lepCorrect/totalPast)
print('1 jet correct', j1Correct/totalPast)
print('lep and 1 jet correct', lep1jCorrect/totalPast)
'''
rightTruthDF = pd.DataFrame(rightTruth)
wrongTruthDF = pd.DataFrame(wrongTruth)
wrongPredDF = pd.DataFrame(wrongPred)

rightTruthDF.to_csv('outputData/rightTruthAllBad.csv', index=False)
wrongTruthDF.to_csv('outputData/wrongTruthAllBad.csv', index=False)
wrongPredDF.to_csv('outputData/wrongPredAllBad.csv', index=False)
'''

plt.figure()
plt.hist(truthScores, 30)
plt.savefig('plots/higgsTopTruthScores.png')

plt.figure()
plt.hist(wrongScores, 30)
plt.savefig('plots/higgsTopRightScores.png')

plt.figure()
plt.hist(wrongScores, 30)
plt.savefig('plots/higgsTopWrongScores.png')

plt.figure()
plt.hist(badScores, 30)
plt.savefig('plots/higgsTopBadScores.png')
