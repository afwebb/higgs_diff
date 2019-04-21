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
#import matplotlib.pyplot as plt

inputFile = sys.argv[1]
modelPath = sys.argv[2]

f=uproot.open(inputFile)
nom=f.get('nominal')

totalPast = 0
nCorrect = 0
lepCorrect = 0
lep1jCorrect = 0

events = []
bestScores = []

wrongTruth = []
wrongPred = []
rightTruth = []

xgbModel = pickle.load(open(modelPath, "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi


la=f['nominal'].lazyarrays(['higgs*','lep_*','jet_*','met','met_*','track_jet_*','truth_jet_*'])

def flatDict(lep, jet1, jet2, met, jet1_MV2c10, jet2_MV2c10):
    k = {}

    k['lep_Pt'] = lep.Pt()
    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj0'] = lep.DeltaR(jet1)
    k['Ptlj0'] = (lep+jet1).Pt()
    k['Mlj0'] = (lep+jet1).M()

    k['dRlj1'] = lep.DeltaR(jet2)
    k['Ptlj1'] = (lep+jet2).Pt()
    k['Mlj1'] = (lep+jet2).M()

    k['dR(jj)(l)'] = (jet1 + jet2).DeltaR(lep + met)

    k['MhiggsCand'] = (jet1+jet2+lep).M()

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    return k


def vecDict(lep, jet1, jet2, met, jet1_MV2c10, jet2_MV2c10):
    q = {}

    #q['match'] = match

    q['lep_Pt_0'] = lep.Pt()
    q['lep_Eta_0'] = lep.Eta()
    phi_0 = lep.Phi()
    q['lep_E_0'] = lep.E()

    q['jet_Pt_0'] = jet1.Pt()
    q['jet_Eta_0'] = jet1.Eta()
    q['jet_Phi_0'] = calc_phi(phi_0, jet1.Phi())
    q['jet_E_0'] = jet1.E()
    q['jet_MV2c10_0'] = jet1_MV2c10

    q['jet_Pt_1'] = jet2.Pt()
    q['jet_Eta_1'] = jet2.Eta()
    q['jet_Phi_1'] = calc_phi(phi_0, jet2.Phi())
    q['jet_E_1'] = jet2.E()
    q['jet_MV2c10_1'] =jet2_MV2c10

    q['MET'] = met.Pt()
    q['MET_phi'] = calc_phi(phi_0, met.Phi())

    return q


def drCheck(eta, phi, truth_eta, truth_phi, cut):
    dr = sqrt( (phi-truth_phi)**2 + (eta-truth_eta)**2 )
    #print('DR', dr, pt/part.pt-1)                                                                                                               
    #print('DR', dr)                                                                                                                             
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
    if idx==10000:
        break

    truthComb = []

    higgCand = LorentzVector()
    
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])
    
    fourVecs['met'] = met
    
    higgCand+=met
        
    lep4Vecs = []
    jet4Vecs = []
    jet4VecsMV2c10 = []

    match = 0
    lepMatch = 0

    for i in range(2):
        lep_pt = la[b'lep_pt'][idx][i]
        lep_eta = la[b'lep_eta'][idx][i]
        lep_phi = la[b'lep_phi'][idx][i]
        lep_E = la[b'lep_E'][idx][i]
        lep_flav = la[b'lep_flavor'][idx][i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        
        lep4Vecs.append(lepVec)

        if la[b'lep_parent'][idx][i]==25:
            truthComb.append(i)
            lepMatch+=1

    for i in range(len(la[b'jet_pt'][idx])):#la[b'selected_jets'][i]:
        
        jet_pt = la[b'jet_pt'][idx][i]
        jet_eta = la[b'jet_eta'][idx][i]
        jet_phi = la[b'jet_phi'][idx][i]
        jet_E = la[b'jet_E'][idx][i]
        jet_flav = la[b'jet_flavor'][idx][i]
        jet_MV2c10 = la[b'jet_MV2c10'][idx][i]
        
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)
        
        jet4Vecs.append(jetVec)
        jet4VecsMV2c10.append(jet_MV2c10)
        
        if la[b'jet_parent'][idx][i]==25:
            truthComb.append(i)
            match+=1
               
    if match!=2 or lepMatch!=1: continue

    combos = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]
                
                k = flatDict( lep4Vecs[l], jet4Vecs[i], jet4Vecs[j], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j] )
                q = vecDict( lep4Vecs[l], jet4Vecs[i], jet4Vecs[j], met, jet4VecsMV2c10[i], jet4VecsMV2c10[j] )

                combos.append([k, comb, q])

        #loop over combinations, score them in the BDT, figure out the best result                                                    
    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    #print(pred[best], best)                                                                                                      

    bestComb = combos[best][1]
    lepMatch = bestComb[0]
    jetMatches = bestComb[1:]

    for c, s in zip(combos, pred):
        if c[1][0]==truthComb[0] and c[1][1] in truthComb[1:] and c[1][2] in truthComb[1:]:
            truthK = {**c[0], **c[2]}
            print('truth score', s)

    predK = {**combos[best][0], **combos[best][2]}

    print(bestComb, truthComb, pred[best])

    totalPast+=1

    if sorted(bestComb)==sorted(truthComb):
        nCorrect+=1
        rightTruth.append(truthK)
    else:
        wrongTruth.append(truthK)
        wrongPred.append(predK)

    if bestComb[0]==truthComb[0]:
        lepCorrect+=1
    if bestComb[1] in truthComb[1:] or bestComb[2] in truthComb[1:]:
        lep1jCorrect+=1

print('events passed', totalPast)
print('percent correct', nCorrect/totalPast)
print('lep correct', lepCorrect/totalPast)
print('lep and 1 jet correct', lep1jCorrect/totalPast)

rightTruthDF = pd.DataFrame(rightTruth)
wrongTruthDF = pd.DataFrame(wrongTruth)
wrongPredDF = pd.DataFrame(wrongPred)

rightTruthDF.to_csv('outputData/rightTruthAllBad.csv', index=False)
wrongTruthDF.to_csv('outputData/wrongTruthAllBad.csv', index=False)
wrongPredDF.to_csv('outputData/wrongPredAllBad.csv', index=False)
