import ROOT
import numpy as np
#import rootpy.io
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
from dict_higgs1l import higgs1lDict
#import matplotlib.pyplot as plt

inf = sys.argv[1]
outDir = sys.argv[2]

f = ROOT.TFile.Open(inf)
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
nom = f.Get('nominal')

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

nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))

    nom.GetEntry(idx)

    if nom.trilep_type==0: continue
    if nom.nJets<2: continue
    if nom.nJets_MV2c10_70==0: continue
    if len(nom.lep_pt)!=3: continue
    if nom.lep_pt[0]<10000: continue
    if nom.lep_pt[1]<20000: continue
    if nom.lep_pt[2]<20000: continue

    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met, 0, nom.met_phi, nom.met)
    
    lepH = []
    lepB = []

    for i in range(3):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(nom.lep_pt[i], nom.lep_eta[i], nom.lep_phi[i], nom.lep_E[i])
        if nom.lep_parent[i]==25:
            lepH.append(lep)
        else:
            lepB.append(lep)

    jets = []
    higgsJets = []
    badJets = []
    for i in range(len(nom.jet_pt)):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(nom.jet_pt[i], nom.jet_eta[i], nom.jet_phi[i], nom.jet_E[i])
        jets.append(jet)
        
        if nom.jet_parent[i]==25:
            higgsJets.append(i)
        else:
            badJets.append(i)
        
    if len(lepH)!=1 or len(higgsJets)!=2: continue

    k = higgs1lDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepH[0], met, 
                     nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[ higgsJets[1] ], lepB[0], lepB[1],
                     nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[ higgsJets[1] ], 
                     1 )
    eventsFlat.append(k)

    for l in range(2):
        k = higgs1lDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepB[0], met, 
                         nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[ higgsJets[1] ], lepH[0], lepB[1],
                         nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[ higgsJets[1] ], 
                         0 )
        eventsFlat.append(k)

    for l in range(2):
        if len(badJets)<2: break
        i,j = random.sample(badJets,2)
        k = higgs1lDict( jets[i], jets[j], lepH[0], met, 
                         nom.jet_MV2c10[i], nom.jet_MV2c10[j], lepB[0], lepB[1],
                         nom.jet_numTrk[i], nom.jet_numTrk[j],
                         0 )
        eventsFlat.append(k)

        k = higgs1lDict( jets[i], jets[ higgsJets[1] ], lepH[0], met, 
                         nom.jet_MV2c10[i], nom.jet_MV2c10[ higgsJets[1]], lepB[0], lepB[1], 
                         nom.jet_numTrk[i], nom.jet_numTrk[ higgsJets[1]],
                         0 )
        eventsFlat.append(k)
        
        k = higgs1lDict( jets[ higgsJets[0] ], jets[j], lepH[0], met, 
                         nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[j], lepB[0], lepB[1],
                         nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[j],
                         0 )
        eventsFlat.append(k)

    for l in range(min([6, len(badJets)])):
        if len(badJets)<2: break
        i,j = random.sample(badJets,2)
        k = higgs1lDict( jets[i], jets[j], lepB[0], met, 
                         nom.jet_MV2c10[i], nom.jet_MV2c10[j], lepH[0], lepB[1],
                         nom.jet_numTrk[i], nom.jet_numTrk[j],
                         0 )
        eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outDir+'/'+dsid+'Flat.csv', index=False)