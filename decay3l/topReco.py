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
from dict_top3l import topDict
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

    if e.trilep_type==0: continue
    if e.nJets<2: continue
    if e.nJets_MV2c10_70==0: continue
    if len(e.lep_pt)!=3: continue
    if e.lep_pt[0]<10000: continue
    if e.lep_pt[1]<20000: continue
    if e.lep_pt[2]<20000: continue

    leps = []
    for i in range(3):
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

    k = topDict( jets[ topJets[0] ], jets[ topJets[1] ], leps[0], leps[1], leps[2], met, 
                 e.jet_MV2c10[ topJets[0] ], e.jet_MV2c10[ topJets[1] ],
                 e.jet_jvt[ topJets[0] ], e.jet_jvt[ topJets[1] ], 
                 e.jet_numTrk[ topJets[0] ], e.jet_numTrk[ topJets[1] ],
                 1 )
    eventsFlat.append(k)

    if len(badJets)>2:

        for l in range(min([3, len(badJets)]) ):
            i,j = random.sample(badJets,2)
            k = topDict( jets[ topJets[0] ], jets[j], leps[0], leps[1], leps[2], met, 
                         e.jet_MV2c10[ topJets[0] ], e.jet_MV2c10[j], 
                         e.jet_jvt[ topJets[0] ], e.jet_jvt[j], 
                         e.jet_numTrk[ topJets[0] ], e.jet_numTrk[j],
                         0 )
            eventsFlat.append(k)

        for l in range(min([3, len(badJets)]) ):
            i,j = random.sample(badJets,2)
            k = topDict( jets[ i ], jets[ topJets[1] ], leps[0], leps[1], leps[2], met, 
                         e.jet_MV2c10[ i ], e.jet_MV2c10[ topJets[1] ], 
                         e.jet_jvt[ i ], e.jet_jvt[ topJets[1] ], 
                         e.jet_numTrk[ i ], e.jet_numTrk[ topJets[1] ],
                         0 )
            eventsFlat.append(k)
            
        for l in range(min([6, len(badJets)]) ):
            i,j = random.sample(badJets,2)
            k = topDict( jets[i], jets[j], leps[0], leps[1], leps[2], met, 
                         e.jet_MV2c10[i], e.jet_MV2c10[j], 
                         e.jet_jvt[i], e.jet_jvt[j], 
                         e.jet_numTrk[i], e.jet_numTrk[j],
                         0 )
            eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('topLepCutFiles/'+dsid+'Flat.csv', index=False)
