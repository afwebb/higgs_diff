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
                           'm_jet*','selected_jets', 'm_truth_jet_pt', 'm_truth_jet_eta', 
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


# In[6]:


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

    if la[b'higgsDecayMode'][idx] != 3: continue
    if la[b'total_leptons'][idx] < 1: continue
    if la[b'dilep_type'][idx] < 1: continue
    if la[b'total_charge'][idx] == 0: continue
    if la[b'nJets_OR_T'][idx] <4: continue
        
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
    
    lepMatch = -1
    for i in range(2):
        
        lepH = LorentzVector()
        lepH.SetPtEtaPhiE(lepPts[i], lepEtas[i], lepPhis[i], lepEs[i])
        
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

    c = make_partdict(la,idx)
    higgsID = 0
    for x in c:
        if c[x].pdgid==25: higgsID = x

    jetCands = []
    jetTest = []
    Ws = c[higgsID].children
    for w in Ws:
        for child in c[w].children:
            if child in c: ch = c[child]
            else: continue
            if abs(ch.pdgid) in range(1,5): jetTest.append(child)
            #elif abs(ch.pdgid)==24: 
        jetCands=[*jetCands, *c[w].children]
        
    if len(jetTest)!=2:
        continue
        
    match=0  
    truthJets = []
    higgsJets=[]
    badJets=[]
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

                jetVec = LorentzVector()
                jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)
                
                dr = sqrt(unwrap([jet_phi-c[j].phi])**2+(jet_eta-c[j].eta)**2)
                if dr<0.3 and jet_flav==abs(c[j].pdgid):
                    truthJets.append(i)
                    higgCand+=jetVec
                    higgsJets.append(jetVec)
                    match+=1
                else:
                    badJets.append(jetVec)
               
    if match!=2: continue

    fourVecs['higgsJets'] = higgsJets
    fourVecs['badJets'] = badJets
    fourVecs['higgsCand'] = higgCand
    
    fourVecDicts.append(fourVecs)
    
    higgVecs.append(higgCand)

import random

eventsFlat = []
eventsVec = []

for f in fourVecDicts:
    k = {}
    
    k['match'] = 1
    
    k['dRjj'] = f['higgsJets'][0].DeltaR(f['higgsJets'][1])
    k['Ptjj'] = (f['higgsJets'][0]+f['higgsJets'][1]).Pt()
    k['Mjj'] = (f['higgsJets'][0]+f['higgsJets'][1]).M()
    
    k['dRlj0'] = f['higgsLep'].DeltaR(f['higgsJets'][0])
    k['Ptlj0'] = (f['higgsLep']+f['higgsJets'][0]).Pt()
    k['Mlj0'] = (f['higgsLep']+f['higgsJets'][0]).M()

    k['dRlj1'] = f['higgsLep'].DeltaR(f['higgsJets'][1])
    k['Ptlj1'] = (f['higgsLep']+f['higgsJets'][1]).Pt()
    k['Mlj1'] = (f['higgsLep']+f['higgsJets'][1]).M()
    
    k['MlepMet'] = (f['higgsLep']+f['met']).M()
    k['dRlepMet'] = f['higgsLep'].DeltaR(f['met'])

    k['dRj0Met'] = f['higgsJets'][0].DeltaR(f['met'])
    k['dRj1Met'] = f['higgsJets'][1].DeltaR(f['met'])

    k['MhiggsCand'] = (f['higgsJets'][0]+f['higgsJets'][1]+f['higgsLep']+f['met']).M()
    
    eventsFlat.append(k)

    q = {}
    
    q['match'] = 1
    
    q['lep_Pt_0'] = f['higgsLep'].Pt()
    q['lep_Eta_0'] = f['higgsLep'].Eta()
    phi_0 = f['higgsLep'].Phi()
    q['lep_E_0'] = f['higgsLep'].E()
    
    q['jet_Pt_0'] = f['higgsJets'][0].Pt()
    q['jet_Eta_0'] = f['higgsJets'][0].Eta()
    q['jet_Phi_0'] = calc_phi(phi_0, f['higgsJets'][0].Phi())
    q['jet_E_0'] = f['higgsJets'][0].E()
    
    q['jet_Pt_1'] = f['higgsJets'][1].Pt()
    q['jet_Eta_1'] = f['higgsJets'][1].Eta()
    q['jet_Phi_1'] = calc_phi(phi_0, f['higgsJets'][1].Phi())
    q['jet_E_1'] = f['higgsJets'][1].E()
    
    q['MET'] = f['met'].Pt()
    q['MET_phi'] = calc_phi(phi_0, f['met'].Phi())
    
    eventsVec.append(q)

for f in fourVecDicts:

    for q in range(3):
        
        k = {}
    
        k['match'] = 0
        
        i,j = random.sample(range(len(f['badJets'])),2)
    
        k['dRjj'] = f['badJets'][i].DeltaR(f['badJets'][j])
        k['Ptjj'] = (f['badJets'][i]+f['badJets'][j]).Pt()
        k['Mjj'] = (f['badJets'][i]+f['badJets'][j]).M()

        k['dRlj0'] = f['badLep'].DeltaR(f['badJets'][i])
        k['Ptlj0'] = (f['badLep']+f['badJets'][i]).Pt()
        k['Mlj0'] = (f['badLep']+f['badJets'][i]).M()

        k['dRlj1'] = f['badLep'].DeltaR(f['badJets'][j])
        k['Ptlj1'] = (f['badLep']+f['badJets'][j]).Pt()
        k['Mlj1'] = (f['badLep']+f['badJets'][j]).M()

        k['MlepMet'] = (f['badLep']+f['met']).M()
        k['dRlepMet'] = f['badLep'].DeltaR(f['met'])

        k['dRj0Met'] = f['badJets'][i].DeltaR(f['met'])
        k['dRj1Met'] = f['badJets'][j].DeltaR(f['met'])

        k['MhiggsCand'] = (f['badJets'][i]+f['badJets'][j]+f['badLep']+f['met']).M()

        eventsFlat.append(k)

        q = {}
        
        q['match'] = 0
        
        q['lep_Pt_0'] = f['badLep'].Pt()
        q['lep_Eta_0'] = f['badLep'].Eta()
        phi_0 = f['badLep'].Phi()
        q['lep_E_0'] = f['badLep'].E()
        
        q['jet_Pt_0'] = f['badJets'][i].Pt()
        q['jet_Eta_0'] = f['badJets'][i].Eta()
        q['jet_Phi_0'] = calc_phi(phi_0, f['badJets'][i].Phi())
        q['jet_E_0'] = f['badJets'][i].E()

        q['jet_Pt_1'] = f['badJets'][j].Pt()
        q['jet_Eta_1'] = f['badJets'][j].Eta()
        q['jet_Phi_1'] = calc_phi(phi_0, f['badJets'][j].Phi())
        q['jet_E_1'] = f['badJets'][j].E()
        
        q['MET'] = f['met'].Pt()
        q['MET_phi'] = calc_phi(phi_0, f['met'].Phi())
        
        eventsVec.append(q)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outputFile+'Flat.csv', index=False)


dfVec = pd.DataFrame.from_dict(eventsVec)
dfVec = shuffle(dfVec)

dfVec.to_csv(outputFile+'Vec.csv', index=False)
