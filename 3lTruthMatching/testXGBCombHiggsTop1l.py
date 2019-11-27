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
from dict_higgsTop1l import higgsTop1lDict
from dict_top3l import topDict
import matplotlib.pyplot as plt

inputFile = sys.argv[1]
modelPath = sys.argv[2]
topPath = sys.argv[3]

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
topModel = pickle.load(open(topPath, "rb"))

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

nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==30000:       
        break                                                                                                                                   

    nom.GetEntry(idx)

    if nom.trilep_type==0: continue
    if nom.nJets<2: continue
    if nom.nJets_MV2c10_70==0: continue
    if len(nom.lep_pt)!=3: continue
    if nom.lep_pt[0]<10000: continue
    if nom.lep_pt[1]<20000: continue
    if nom.lep_pt[2]<20000: continue

    truthComb = []
    match=0
    lepMatch = 0

    higgCand = LorentzVector()
    
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met, 0, nom.met_phi, nom.met)
    
    lep4Vecs = []
    for i in range(3):
        lep_pt = nom.lep_pt[i]
        lep_eta = nom.lep_eta[i]
        lep_phi = nom.lep_phi[i]
        lep_E = nom.lep_E[i]
        lep_flav = nom.lep_flavor[i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)

        lep4Vecs.append(lepVec)

        if nom.lep_parent[i]==25:
            truthComb.append(i)
            lepMatch+=1

    jet4Vecs = []
    jet4VecsMV2c10 = []
    for i in range(len(nom.jet_pt)):
        #if nom.jet_jvt[i]<0.59:
        #    continue

        jet_pt = nom.jet_pt[i]
        jet_eta = nom.jet_eta[i]
        jet_phi = nom.jet_phi[i]
        jet_E = nom.jet_E[i]
        jet_flav = nom.jet_flavor[i]
        jet_MV2c10 = nom.jet_MV2c10[i]

        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)

        jet4Vecs.append(jetVec)
        jet4VecsMV2c10.append(jet_MV2c10)

        if abs(nom.jet_parent[i])==25:
            truthComb.append(i)
            match+=1

    if match!=2 or lepMatch!=1: continue

    if len(jet4Vecs)<4:
        continue

    combosTop = []

    for i in range(len(jet4Vecs)-1):
        for j in range(i+1, len(jet4Vecs)):
            comb = [i,j]
            t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met,
                         jet4VecsMV2c10[i], jet4VecsMV2c10[j],
                         nom.jet_numTrk[i], nom.jet_numTrk[j])
            combosTop.append([t, comb])

    dfTop = pd.DataFrame.from_dict([x[0] for x in combosTop])
    xgbMatTop = xgb.DMatrix(dfTop, feature_names=list(dfTop))

    pred = topModel.predict(xgbMatTop)
    best = np.argmax(pred)
    bestCombTop = combosTop[best][1]

    top1 = jet4Vecs[bestCombTop[0]]
    top2 = jet4Vecs[bestCombTop[1]]
    topScore = pred[best]

    combos = []

    for l in range(1, len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]

                if i in bestCombTop or j in bestCombTop:
                    continue

                if l==1:
                    k = higgsTop1lDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, 
                                        jet4VecsMV2c10[i], jet4VecsMV2c10[j], lep4Vecs[0], lep4Vecs[2],
                                        nom.jet_numTrk[i], nom.jet_numTrk[j], 
                                        top1, top2, topScore)
                else:
                    k = higgsTop1lDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, 
                                        jet4VecsMV2c10[i], jet4VecsMV2c10[j], lep4Vecs[0], lep4Vecs[1],
                                        nom.jet_numTrk[i], nom.jet_numTrk[j],
                                        top1, top2, topScore)

                combos.append([k, comb])

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    lepMatch = bestComb[0]
    jetMatches = bestComb[1:]

    #for c, s in zip(combos, pred):
    #    if c[1][0]==truthComb[0] and c[1][1] in truthComb[1:] and c[1][2] in truthComb[1:]:
    #        #truthK = {**c[0], **c[2]}
    #        print('truth score', s, c[1], truthComb)

    #predK = {**combos[best][0], **combos[best][2]}

    #print(bestComb, truthComb, pred[best])

    totalPast+=1

    if sorted(bestComb)==sorted(truthComb):
        nCorrect+=1
        rightScores.append(pred[best])
        #rightTruth.append(truthK)
    else:
        wrongScores.append(pred[best])
        #wrongTruth.append(truthK)
        #wrongPred.append(predK)

    if bestComb[0]==truthComb[0]:
        lepCorrect+=1
    if bestComb[1] in truthComb[1:] and bestComb[2] in truthComb[1:]:
        j1Correct+=1
    if bestComb[0]==truthComb[0] and (bestComb[1] in truthComb[1:] or bestComb[2] in truthComb[1:]):
        lep1jCorrect+=1

print('events passed', totalPast)
print('percent correct', nCorrect/totalPast)
print('lep correct', lepCorrect/totalPast)
print('both jets correct', j1Correct/totalPast)
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
plt.savefig('plots/higgs1lTruthScores.png')

plt.figure()
plt.hist(rightScores, 30, alpha=0.5, label='right')
plt.hist(wrongScores, 30, alpha=0.5, label='wrong')
plt.legend()
plt.savefig('plots/higgs1lMatchScores.png')


