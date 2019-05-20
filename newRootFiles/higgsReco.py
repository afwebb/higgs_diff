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
from dict_higgs import higgsDict
#import matplotlib.pyplot as plt

inf = sys.argv[1]
outDir = sys.argv[2]

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

    k = higgsDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepH[0], met, e.jet_MV2c10[ higgsJets[0] ], e.jet_MV2c10[ higgsJets[1] ], lepB[0], 1 )
    eventsFlat.append(k)

    for l in range(2):
        k = higgsDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepB[0], met, e.jet_MV2c10[ higgsJets[0] ], e.jet_MV2c10[ higgsJets[1] ], lepH[0], 0 )
        eventsFlat.append(k)

    for l in range(2):
        i,j = random.sample(badJets,2)
        k = higgsDict( jets[i], jets[j], lepH[0], met, e.jet_MV2c10[i], e.jet_MV2c10[j], lepB[0], 0 )
        eventsFlat.append(k)

        k = higgsDict( jets[i], jets[ higgsJets[1] ], lepH[0], met, e.jet_MV2c10[i], e.jet_MV2c10[ higgsJets[1]], lepB[0], 0 )
        eventsFlat.append(k)
        
        k = higgsDict( jets[ higgsJets[0] ], jets[j], lepH[0], met, e.jet_MV2c10[ higgsJets[0] ], e.jet_MV2c10[j], lepB[0], 0 )
        eventsFlat.append(k)

    for l in range(min([6, len(badJets)])):
        i,j = random.sample(badJets,2)
        k = higgsDict( jets[i], jets[j], lepB[0], met, e.jet_MV2c10[i], e.jet_MV2c10[j], lepH[0], 0 )
        eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outDir+'/'+dsid+'Flat.csv', index=False)
