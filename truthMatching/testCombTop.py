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


la=f['nominal'].lazyarrays(['m_truth*','dilep_type','higgs*','lep_Pt*','lep_Eta_*', 'lep_E_*','total_charge',
                            'total_leptons',
                           'lep_Phi*','lep_ID*','lep_Index*',
                            #'electron_eta','electron_phi','electron_pt',
                            #'muon_eta','muon_phi','muon_pt',
                           'm_jet*','selected_jets', 'm_truth_jet_pt', 'm_truth_jet_eta', 
                            'm_truth_jet_phi', 'm_truth_jet_m', 'm_truth_jet_Hcount', 'm_truth_jet_Tcount','nJets_OR_T',
                            'MET_RefFinal_et', 'MET_RefFinal_phi', 'lep_jet_nTrk_0', 'lep_jet_nTrk_1', 'm_jet_numTrk'])


# In[5]:


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
def flatDict( lep, jet1, jet2, met, jet1_MV2c10, jet2_MV2c10):
    k = {}

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    k['dRlj0'] = lep.DeltaR(jet1)
    k['Ptlj0'] = (lep+jet1).Pt()
    k['Mlj0'] = (lep+jet1).M()

    k['dRlj1'] = lep.DeltaR(jet2)
    k['Ptlj1'] = (lep+jet2).Pt()
    k['Mlj1'] = (lep+jet2).M()

    k['MlepMet'] = (lep+met).M()
    k['dRlepMet'] = lep.DeltaR(met)

    k['Mj0Met'] = (jet1+met).M()
    k['dRj0Met'] = jet1.DeltaR(met)

    k['Mj1Met'] = (jet2+met).M()
    k['dRj1Met'] = jet2.DeltaR(met)

    k['dR(jj)(lmet)'] = (jet1 + jet2).DeltaR(lep + met)

    k['MhiggsCand'] = (jet1+jet2+lep+met).M()
    
    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    return k
'''

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


from collections import namedtuple
jettype = namedtuple('jettype', ['pdgid','eta', 'phi', 'pt', 'E'])

def make_jetdict(la, idx):
    rv = dict(zip([x for x in range(len(la[b'm_jet_pt'][idx]))],
                 (jettype(*_2) for
                  _2 in zip(la[b'm_jet_flavor_truth_label_ghost'][idx],
                            la[b'm_jet_eta'][idx],
                            la[b'm_jet_phi'][idx],
                            la[b'm_jet_pt'][idx],
                            la[b'm_jet_E'][idx])
                 )
                 ))
    return rv


# In[7]:


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

for idx in range(len(la[b'nJets_OR_T']) ):
    #current+=1
    if idx%10000==0:
        print(idx)                                                                                                                              
    if idx==100000:
        break

    if la[b'higgsDecayMode'][idx] != 3: continue
    if la[b'total_leptons'][idx] < 1: continue
    if la[b'dilep_type'][idx] < 1: continue
    if la[b'total_charge'][idx] == 0: continue
    #if la[b'nJets_OR_T'][idx] <4: continue
        
    truthComb = []

    higgCand = LorentzVector()
    
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'MET_RefFinal_et'][idx], 0, la[b'MET_RefFinal_phi'][idx], la[b'MET_RefFinal_et'][idx])
    
    fourVecs['met'] = met
    
    higgCand+=met
        
    truth_dict = make_partdict(la, idx)
    jet_dict = make_jetdict(la,idx)

    lepPts = [la[b'lep_Pt_0'][idx], la[b'lep_Pt_1'][idx]]
    lepEtas = [la[b'lep_Eta_0'][idx], la[b'lep_Eta_1'][idx]]
    lepPhis = [la[b'lep_Phi_0'][idx], la[b'lep_Phi_1'][idx]]
    lepEs = [la[b'lep_E_0'][idx], la[b'lep_E_1'][idx]]
    lepIDs = [la[b'lep_ID_0'][idx], la[b'lep_ID_1'][idx]]

    lep4Vecs = []

    lepMatch = -1
    for i in range(2):
        
        lepH = LorentzVector()
        lepH.SetPtEtaPhiE(lepPts[i], lepEtas[i], lepPhis[i], lepEs[i])
        
        lep4Vecs.append(lepH)

        for a in truth_dict:
            if abs(truth_dict[a].pdgid) == abs(lepIDs[i]):
                if drCheck(lepEtas[i], lepPhis[i], truth_dict[a].eta, truth_dict[a].phi, 0.01):

                    p = truth_dict[a].parents[0]
                    terminal = False
                    while not terminal:
                        if p in truth_dict:
                            a = p
                            try:
                                p = truth_dict[a].parents[0]
                            except:
                                terminal = True
                        else: terminal = True
                    if truth_dict[a].pdgid == 25: 
                        higgCand+=lepH
                        lepMatch = i
                        fourVecs['higgsLep'] = lepH
            if lepMatch!=i: 
                fourVecs['badLep'] = lepH

    if lepMatch == -1:
        continue

    truthComb.append(lepMatch)

    c = make_partdict(la,idx)
    higgsID = 0
    for x in c:
        if c[x].pdgid==25: 
            higgsID = x

    jetCands = []
    jetTest = []
    Ws = c[higgsID].children
    for w in Ws:
        try:
            if 24 in [abs(c[x].pdgid) for x in c[w].children]:
                childCand = []
                for wChild in c[w].children:
                    if c[wChild].pdgid in [-24, 24]:
                        for x in c[wChild].children:
                            childCand.append(x)
            else:
                childCand = c[w].children
        except:
            childCand = c[w].children

        for child in childCand:
            if child in c: 
                ch = c[child]
            else: 
                continue
            if abs(ch.pdgid) in range(1,5): 
                jetTest.append(child)

        jetCands=[*jetCands, *childCand]

    if len(jetTest)!=2:
        continue
    ''''
    lepMatch = -1
    
    for i in range(2):

        lepH = LorentzVector()
        lepH.SetPtEtaPhiE(lepPts[i], lepEtas[i], lepPhis[i], lepEs[i])

        for l in jetCands:
            if l not in c:
                print('not found')
                continue
            if abs(c[l].pdgid) not in [11, 13]:
                continue
            
            if abs(c[l].pdgid) == abs(lepIDs[i]):
                if drCheck(lepEtas[i], lepPhis[i], c[l].eta, c[l].phi, 0.1): 
                    higgCand+=lepH
                    lepMatch = i
                    fourVecs['higgsLep'] = lepH
        if lepMatch!=i: 
            fourVecs['badLep'] = lepH

    if lepMatch == -1:
        continue
    '''
    match=0  
    truthJets = []

    higgsJets=[]
    higgsJetsMV2c10=[]

    badJets=[]
    badJetsMV2c10=[]

    jet4Vecs = []
    jet4VecsMV2c10 = []

    for j in jetCands:
        if j not in c:
            print('not found')
            continue
        if abs(c[j].pdgid) not in range(1, 5):
            continue
        else:
            #match = False
            for i in range(len(la[b'm_jet_pt'][idx])):#la[b'selected_jets'][i]:

                jet_pt = la[b'm_jet_pt'][idx][i]
                jet_eta = la[b'm_jet_eta'][idx][i]
                jet_phi = la[b'm_jet_phi'][idx][i]
                jet_E = la[b'm_jet_E'][idx][i]
                jet_flav = la[b'm_jet_flavor_truth_label_ghost'][idx][i]
                jet_MV2c10 = la[b'm_jet_flavor_weight_MV2c10'][idx][i]

                jetVec = LorentzVector()
                jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)
                
                dr = sqrt(unwrap([jet_phi-c[j].phi])**2+(jet_eta-c[j].eta)**2)

                jet4Vecs.append(jetVec)
                jet4VecsMV2c10.append(jet_MV2c10)

                if dr<0.3 and jet_flav==abs(c[j].pdgid):
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

    if truthComb[0]==0:
        lepTrack = la[b'lep_jet_nTrk_0'][idx]
    else:
        lepTrack = la[b'lep_jet_nTrk_0'][idx]

    jetTrack1 = la[b'm_jet_pt'][idx][truthComb[1]]
    jetTrack2 = la[b'm_jet_pt'][idx][truthComb[2]]

    print(bestComb, truthComb, pred[best], jetTrack1, jetTrack2)

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
