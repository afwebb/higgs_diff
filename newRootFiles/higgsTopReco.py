import ROOT
import pandas as pd
import numpy as np
import uproot
#import rootpy.io
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
from dict_higgsTop import higgsTopDict
#import matplotlib.pyplot as plt

inf = sys.argv[1]
#outputFile = sys.argv[2]

f = uproot.open(inf)
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

la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*'])

current = 0
nMatch = 0
higgVecs = []

goodMatches = 0
badMatches = 0

fourVecDicts = []
eventsFlat = []
current = 0

for idx in range(len(la[b'met']) ):
    current+=1
    if current%10000==0:
        print(current)

    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])
    
    if len(la[b'lep_pt'][idx])!=2: continue

    lepH = []
    lepB = []

    for i in range(2):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(la[b'lep_pt'][idx][i], la[b'lep_eta'][idx][i], la[b'lep_phi'][idx][i], la[b'lep_E'][idx][i])
        if la[b'lep_parent'][idx][i]==25:
            lepH.append(lep)
        else:
            lepB.append(lep)

    jets = []
    higgsJets = []
    topJets = []
    badJets = []
    for i in range(len(la[b'jet_pt'][idx])):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(la[b'jet_pt'][idx][i], la[b'jet_eta'][idx][i], la[b'jet_phi'][idx][i], la[b'jet_E'][idx][i])
        jets.append(jet)
        
        if la[b'jet_parent'][idx][i]==25:
            higgsJets.append(i)
        elif abs(la[b'jet_parent'][idx][i])==6:
            topJets.append(i)
        else:
            badJets.append(i)
        
    if len(lepH)!=1 or len(higgsJets)!=2 or len(topJets)!=2: continue

    k = higgsTopDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepH[0], met, la[b'jet_MV2c10'][idx][ higgsJets[0] ], la[b'jet_MV2c10'][idx][ higgsJets[1] ], jets[ topJets[0] ], jets[ topJets[1] ], lepB[0], 1 )
    eventsFlat.append(k)

    for l in range(2):
        k = higgsTopDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepB[0], met, la[b'jet_MV2c10'][idx][ higgsJets[0] ], la[b'jet_MV2c10'][idx][ higgsJets[1] ], jets[ topJets[0] ], jets[ topJets[1] ], lepH[0], 0 )
        eventsFlat.append(k)

    if len(badJets) > 2:
        for l in range(2):
            i,j = random.sample(badJets,2)
            k = higgsTopDict( jets[i], jets[j], lepH[0], met, la[b'jet_MV2c10'][idx][i], la[b'jet_MV2c10'][idx][j], jets[ topJets[0] ], jets[ topJets[1] ], lepB[0], 0 )
            eventsFlat.append(k)

        for l in range(min([8, len(badJets)])):
            i,j = random.sample(badJets,2)
            k = higgsTopDict( jets[i], jets[j], lepB[0], met, la[b'jet_MV2c10'][idx][i], la[b'jet_MV2c10'][idx][j], jets[ topJets[0] ], jets[ topJets[1] ], lepH[0], 0 )
            eventsFlat.append(k)

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('higgsTopPflowFiles/'+dsid+'Flat.csv', index=False)
