#import ROOT
import numpy as np
import rootpy.io
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
#import matplotlib.pyplot as plt

inf = sys.argv[1]
#outputFile = sys.argv[2]

f = rootpy.io.root_open(inf)
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
nom = f.get('nominal')

events = []

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi


def flatDict(match, jet1, jet2, lep, met, jet1_MV2c10, jet2_MV2c10):
    k = {}

    k['match'] = match

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

current = 0
nMatch = 0
higgVecs = []

goodMatches = 0
badMatches = 0

fourVecDicts = []
eventsFlat = []
current = 0

for e in nom:
    current+=1
    if current%10000==0:
        print(current)

    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(e.met, 0, e.met_phi, e.met)
    
    if len(e.lep_pt)!=2: continue

    lepH = []
    lepB = []

    for i in range(2):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(e.lep_pt[i], e.lep_eta[i], e.lep_phi[i], e.lep_E[i])
        if e.lep_parent[i]==25:
            lepH.append(lep)
        else:
            lepB.append(lep)

    jets = []
    higgsJets = []
    badJets = []
    for i in range(len(e.jet_pt)):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(e.jet_pt[i], e.jet_eta[i], e.jet_phi[i], e.jet_E[i])
        jets.append(jet)
        
        if e.jet_parent[i]==25:
            higgsJets.append(i)
        else:
            badJets.append(i)
        
    if len(lepH)!=1 or len(higgsJets)!=2: continue

    k = flatDict( 1, jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepH[0], met, e.jet_MV2c10[ higgsJets[0] ], e.jet_MV2c10[ higgsJets[0] ] )
    eventsFlat.append(k)

    for l in range(2):
        k = flatDict( 0, jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepB[0], met, e.jet_MV2c10[ higgsJets[0] ], e.jet_MV2c10[ higgsJets[0] ] )
        eventsFlat.append(k)

    for l in range(2):
        i,j = random.sample(badJets,2)
        k = flatDict( 0, jets[i], jets[j], lepH[0], met, e.jet_MV2c10[i], e.jet_MV2c10[j] )
        eventsFlat.append(k)

    for l in range(8):
        i,j = random.sample(badJets,2)
        k = flatDict( 0, jets[i], jets[j], lepB[0], met, e.jet_MV2c10[i], e.jet_MV2c10[j] )
        eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('higgsFiles/'+dsid+'Flat.csv', index=False)
