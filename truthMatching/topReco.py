import ROOT
import numpy as np
import uproot
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
#import matplotlib.pyplot as plt

inputFile = sys.argv[1]
outputFile = sys.argv[2]

f=uproot.open(inputFile)
nom=f.get('nominal')

events = []

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
                            'm_track_jet*', 'm_jet*','selected_jets', 'm_truth_jet_pt', 'm_truth_jet_eta', 
                            'm_truth_jet_phi', 'm_truth_jet_m', 'm_truth_jet_Hcount', 'm_truth_jet_Tcount','nJets_OR_T',
                            'MET_RefFinal_et', 'MET_RefFinal_phi'])


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


def flatDict(match, jet1, jet2, met, jet1_MV2c10, jet2_MV2c10):
    k = {}

    k['match'] = match

    k['jet_Pt_0'] = jet1.Pt()
    k['jet_Pt_1'] = jet2.Pt()

    k['dRjj'] = jet1.DeltaR(jet2)
    k['Ptjj'] = (jet1+jet2).Pt()
    k['Mjj'] = (jet1+jet2).M()

    #k['MlepMet'] = (lep+met).M()
    #k['dRlepMet'] = lep.DeltaR(met)

    #k['Mj0Met'] = (jet1+met).M()
    #k['dRj0Met'] = jet1.DeltaR(met)

    #k['Mj1Met'] = (jet2+met).M()
    #k['dRj1Met'] = jet2.DeltaR(met)

    k['jet_MV2c10_0'] =jet1_MV2c10
    k['jet_MV2c10_1'] =jet2_MV2c10

    return k

def vecDict(match, jet1, jet2, met, jet1_MV2c10, jet2_MV2c10):
    q = {}

    q['match'] = match

    q['jet_Pt_0'] = jet1.Pt()
    q['jet_Eta_0'] = jet1.Eta()
    q['jet_Phi_0'] = jet1.Phi()
    q['jet_E_0'] = jet1.E()
    q['jet_MV2c10_0'] = jet1_MV2c10

    q['jet_Pt_1'] = jet2.Pt()
    q['jet_Eta_1'] = jet2.Eta()
    q['jet_Phi_1'] = jet2.Phi()
    q['jet_E_1'] = jet2.E()
    q['jet_MV2c10_1'] =jet2_MV2c10

    q['MET'] = met.Pt()
    q['MET_phi'] = met.Phi()

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

goodMatches = 0
badMatches = 0

fourVecDicts = []

for idx in range(len(la[b'nJets_OR_T']) ):
    #current+=1
    if idx%10000==0:
        print(idx)                                                                                                                              
    #if idx==10000:
    #    break

    if la[b'higgsDecayMode'][idx] != 3: continue
    if la[b'total_leptons'][idx] < 1: continue
    if la[b'dilep_type'][idx] < 1: continue
    if la[b'total_charge'][idx] == 0: continue
    
    if la[b'nJets_OR_T'][idx] <4: continue
        
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'MET_RefFinal_et'][idx], 0, la[b'MET_RefFinal_phi'][idx], la[b'MET_RefFinal_et'][idx])
    
    fourVecs['met'] = met
    
    truth_dict = make_partdict(la, idx)
    jet_dict = make_jetdict(la,idx)

    c = make_partdict(la,idx)
    topID = []
    for x in c:
        if abs(c[x].pdgid)==6: topID.append(x)

    jetCands = []
    jetTest = []
    if len(topID)!=2:
        print('two tops not found')
        continue

    Ws = [*c[topID[0]].children, *c[topID[1]].children]

    for w in Ws:
        if abs(c[w].pdgid)==5:
            jetCands.append(w)

    if len(jetCands)!=2:
        print('wrong jets', len(jetCands))
        continue

    match=0  
    truthJets = []

    topJets=[]
    topJetsMV2c10=[]

    badJets=[]
    badJetsMV2c10=[]

    for j in jetCands:
        if j not in c:
            print('not found')
            continue
        if abs(c[j].pdgid)!=5:
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
                if dr<0.3 and jet_flav==abs(c[j].pdgid):
                    truthJets.append(i)
                    topJets.append(jetVec)
                    topJetsMV2c10.append(jet_MV2c10)
                    match+=1
                #else:
                badJets.append(jetVec)
                badJetsMV2c10.append(jet_MV2c10)

    if match==2:
        goodMatches+=1
    else:
        badMatches+=1
        continue

    fourVecs['truthJets'] = truthJets
    fourVecs['topJets'] = topJets
    fourVecs['badJets'] = badJets
    fourVecs['topJetsMV2c10'] = topJetsMV2c10
    fourVecs['badJetsMV2c10'] = badJetsMV2c10

    fourVecDicts.append(fourVecs)
    
import random

eventsFlat = []
eventsVec = []

for f in fourVecDicts:
    
    k = flatDict( 1, f['topJets'][0], f['topJets'][1], f['met'], f['topJetsMV2c10'][0], f['topJetsMV2c10'][1])
    eventsFlat.append(k)

    q = vecDict( 1, f['topJets'][0], f['topJets'][1], f['met'], f['topJetsMV2c10'][0], f['topJetsMV2c10'][1])
    eventsVec.append(q)

for f in fourVecDicts:
    '''
    i,j = random.sample(range(len(f['badJets'])),2)

    k = flatDict( 0, f['topLep'], f['badJets'][i], f['badJets'][j], f['met'], f['badJetsMV2c10'][i], f['badJetsMV2c10'][j] )
    eventsFlat.append(k)

    q = vecDict( 0, f['topLep'], f['badJets'][i], f['badJets'][j], f['met'], f['badJetsMV2c10'][i], f['badJetsMV2c10'][j] )
    eventsVec.append(q)

    if j%2==0:
        k = flatDict( 0, f['badLep'], f['topJets'][i%2], f['badJets'][j], f['met'], f['topJetsMV2c10'][i%2], f['badJetsMV2c10'][j] )
        eventsFlat.append(k)
        
        q = vecDict( 0, f['badLep'], f['topJets'][i%2], f['badJets'][j], f['met'], f['topJetsMV2c10'][i%2], f['badJetsMV2c10'][j] )
        eventsVec.append(q)

    else:
        k = flatDict( 0, f['badLep'], f['badJets'][i], f['topJets'][i%2], f['met'], f['badJetsMV2c10'][i], f['topJetsMV2c10'][i%2] )
        eventsFlat.append(k)
        
        q = vecDict( 0, f['badLep'], f['badJets'][i], f['topJets'][i%2], f['met'], f['badJetsMV2c10'][i], f['topJetsMV2c10'][i%2] )
        eventsVec.append(q)
    '''
    for q in range(4):
        
        i,j = random.sample(range(len(f['badJets'])),2)

        

        k = flatDict( 0, f['badJets'][i], f['badJets'][j], f['met'], f['badJetsMV2c10'][i], f['badJetsMV2c10'][j] )
        eventsFlat.append(k)

        q = vecDict( 0, f['badJets'][i], f['badJets'][j], f['met'], f['badJetsMV2c10'][i], f['badJetsMV2c10'][j] )
        eventsVec.append(q)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outputFile+'Flat.csv', index=False)


dfVec = pd.DataFrame.from_dict(eventsVec)
dfVec = shuffle(dfVec)

dfVec.to_csv(outputFile+'Vec.csv', index=False)


totM = goodMatches+badMatches
print('Good Matches', goodMatches/totM)
print('Bad Matches', badMatches/totM)

