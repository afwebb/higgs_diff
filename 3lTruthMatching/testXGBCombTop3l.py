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
from dict_top3l import topDict
import matplotlib.pyplot as plt

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

        if abs(nom.jet_parent[i])==6:
            truthComb.append(i)
            match+=1

    if match!=2: continue

    combos = []

    for i in range(len(jet4Vecs)-1):
        for j in range(i+1, len(jet4Vecs)):
            comb = [i,j]

            #if nom.jet_jvt[i]<0.59 or nom.jet_jvt[j]<0.59:
            #    continue

            t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met,
                         jet4VecsMV2c10[i], jet4VecsMV2c10[j],
                         #nom.jet_jvt[i], nom.jet_jvt[j],
                         nom.jet_numTrk[i], nom.jet_numTrk[j])

    
            combos.append([t, comb])

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    jetMatches = bestComb

    for c, s in zip(combos, pred):
        if c[1][0] in truthComb and c[1][1] in truthComb:
            #truthK = {**c[0], **c[1]}
            truthScores.append(s)
            #print('truth score', s)

    #predK = {**combos[best][0], **combos[best][1]}

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

'''
plt.figure()
plt.hist(truthScores, 30)
plt.savefig('plots/topNoJVTTruthScores.png')

plt.figure()
plt.hist(rightScores, 30, alpha=0.5, label='right')
plt.hist(wrongScores, 30, alpha=0.5, label='wrong')
plt.legend()
plt.savefig('plots/topNoJVTMatchScores.png')


