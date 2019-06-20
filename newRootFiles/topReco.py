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
import uproot
from dict_top import topDict
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

current = 0
nMatch = 0
higgVecs = []

goodMatches = 0
badMatches = 0

fourVecDicts = []
eventsFlat = []
current = 0

la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*',
                            'nJets*', 'total_*', 'dilep*', 'nJets_MV2c10_70' ])


totalEvt = len(la[b'met'])

for idx in range(len(la[b'met']) ):
    current+=1
    if current%10000==0:
        print(str(current)+'/'+str(totalEvt))
    
    if la[b'total_leptons'][idx]!=2: continue
    if la[b'total_charge'][idx]==0: continue
    if la[b'dilep_type'][idx]<1: continue
    if la[b'nJets'][idx]<4: continue
    if la[b'nJets_MV2c10_70'][idx]<1: continue

    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])
    
    #if len(la[b'lep_pt'][idx])!=2: continue

    leps = []
    for i in range(2):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(la[b'lep_pt'][idx][i], la[b'lep_eta'][idx][i], la[b'lep_phi'][idx][i], la[b'lep_E'][idx][i])
        leps.append(lep)

    jets = []
    topJets = []
    badJets = []
    for i in range(len(la[b'jet_pt'][idx])):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(la[b'jet_pt'][idx][i], la[b'jet_eta'][idx][i], la[b'jet_phi'][idx][i], la[b'jet_E'][idx][i])
        jets.append(jet)
        
        if la[b'jet_parent'][idx][i]==6:
            topJets.append(i)
        else:
            badJets.append(i)
        
    if len(topJets)!=2: continue

    k = topDict( jets[ topJets[0] ], jets[ topJets[1] ], leps[0], leps[1], met, 
                 la[b'jet_MV2c10'][idx][ topJets[0] ], la[b'jet_MV2c10'][idx][ topJets[1] ], 
                 la[b'jet_jvt'][idx][ topJets[0] ], la[b'jet_jvt'][idx][ topJets[1] ],
                 la[b'jet_numTrk'][idx][ topJets[0] ], la[b'jet_numTrk'][idx][ topJets[1] ],
                 1 )
    eventsFlat.append(k)

    for l in range(min([3, len(badJets)]) ):
        i,j = random.sample(badJets,2)
        k = topDict( jets[ topJets[0] ], jets[j], leps[0], leps[1], met, 
                     la[b'jet_MV2c10'][idx][ topJets[0] ], la[b'jet_MV2c10'][idx][j], 
                     la[b'jet_jvt'][idx][ topJets[0] ], la[b'jet_jvt'][idx][j],
                     la[b'jet_numTrk'][idx][ topJets[0] ], la[b'jet_numTrk'][idx][j],
                     0 )
        eventsFlat.append(k)

    for l in range(min([3, len(badJets)]) ):
        i,j = random.sample(badJets,2)
        k = topDict( jets[ i ], jets[ topJets[1] ], leps[0], leps[1], met, 
                     la[b'jet_MV2c10'][idx][ i ], la[b'jet_MV2c10'][idx][ topJets[1] ], 
                     la[b'jet_jvt'][idx][ i ], la[b'jet_jvt'][idx][ topJets[1] ],
                     la[b'jet_numTrk'][idx][ i ], la[b'jet_numTrk'][idx][ topJets[1] ],
                     0 )
        eventsFlat.append(k)

    for l in range(min([6, len(badJets)]) ):
        i,j = random.sample(badJets,2)
        k = topDict( jets[i], jets[j], leps[0], leps[1], met, 
                     la[b'jet_MV2c10'][idx][i], la[b'jet_MV2c10'][idx][j], 
                     la[b'jet_jvt'][idx][i], la[b'jet_jvt'][idx][j],
                     la[b'jet_numTrk'][idx][i], la[b'jet_numTrk'][idx][j],
                     0 )
        eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('topPflowFiles/'+dsid+'Flat.csv', index=False)
