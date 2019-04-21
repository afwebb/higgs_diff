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


def flatDict(match, jet1, jet2, lep1, lep2, met, jet1_MV2c10, jet2_MV2c10):
    k = {}

    k['match'] = match

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

    leps = []
    for i in range(2):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(e.lep_pt[i], e.lep_eta[i], e.lep_phi[i], e.lep_E[i])
        leps.append(lep)

    jets = []
    topJets = []
    badJets = []
    for i in range(len(e.jet_pt)):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(e.jet_pt[i], e.jet_eta[i], e.jet_phi[i], e.jet_E[i])
        jets.append(jet)
        
        if e.jet_parent[i]==6:
            topJets.append(i)
        else:
            badJets.append(i)
        
    if len(topJets)!=2: continue

    k = flatDict( 1, jets[ topJets[0] ], jets[ topJets[1] ], leps[0], leps[1], met, e.jet_MV2c10[ topJets[0] ], e.jet_MV2c10[ topJets[0] ] )
    eventsFlat.append(k)

    for l in range(min([8, len(badJets)]):
        i,j = random.sample(badJets,2)
        k = flatDict( 0, jets[i], jets[j], leps[0], leps[1], met, e.jet_MV2c10[i], e.jet_MV2c10[j] )
        eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('topFiles/'+dsid+'Flat.csv', index=False)
