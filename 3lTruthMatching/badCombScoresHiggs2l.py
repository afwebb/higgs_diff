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

'''
from collections import namedtuple
parttype = namedtuple('parttype', ['barcode', 'pdgid', 'status', 'eta', 'phi', 'pt', 'parents', 'children'])

def make_partdict(la, idx):
    rv = dict(zip(la[b'm_truth_barcode'][idx],
                 (parttype(*_2) for
                  _2 in zip(la[b'm_truth_barcode'][idx],
                            la[b'm_truth_pdgId'][idx],
                            la[b'm_truth_status'][idx],
                            la[b'm_truth_eta'][idx],
                            la[b'm_truth_phi'][idx],
                            la[b'm_truth_pt'][idx],
                            la[b'm_truth_parents'][idx],
                            la[b'm_truth_children'][idx])
                 )
                 ))
    return rv

'''
'''
def higgsDict( jet1, jet2, lep, met, jet1_MV2c10, jet2_MV2c10, lepO):
    k = {}

    #k['match'] = match

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

    k['lep_Pt_Other'] = lepO.Pt()
    k['dR(jj)(lepOther)'] = (jet1+jet2).DeltaR(lepO)
    k['Mj0lO'] = (jet1+lepO).M()
    k['Mj1lO'] = (jet2+lepO).M()

    return k
'''
def topDict( jet1, jet2, lep1, lep2, met, jet1_MV2c10, jet2_MV2c10):
    k = {}

    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj00'] = lep1.DeltaR(jet1)
    k['Mlj00'] = (lep1+jet1).M()

    k['dRlj01'] = lep1.DeltaR(jet2)
    k['Mlj01'] = (lep1+jet2).M()

    k['dRlj10'] = lep2.DeltaR(jet1)
    k['Mlj10'] = (lep2+jet1).M()

    k['dRlj11'] = lep2.DeltaR(jet2)
    k['Mlj11'] = (lep2+jet2).M()

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

    lepMatch = []
    badLep = []
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
            badLep.append(i)

    if len(lepMatch)!=1:
        continue

    truthComb=lepMatch

    combos = []

    for i in range(len(lep4Vecs)-1):
        for j in range(i+1, len(lep4Vecs)):
            comb = [i,j]
            k = higgsDict( lep4Vecs[i], lep4Vecs[j], lep4Vecs[badLep], met)
            combos.append([k, comb])

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

    if sorted(bestComb)==sorted(truthComb):
        nCorrect+=1
        rightScores.append(pred[best])
        #rightTruth.append(truthK)
    else:
        wrongScores.append(pred[best])
        #wrongTruth.append(truthK)
        #wrongPred.append(predK)

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


plt.figure()
plt.hist(truthScores, 30)
plt.savefig('plots/higgs2lTruthScores.png')

plt.figure()
plt.hist(wrongScores, 30)
plt.savefig('plots/higgs2lRightScores.png')

plt.figure()
plt.hist(wrongScores, 30)
plt.savefig('plots/higgs2lWrongScores.png')
'''
