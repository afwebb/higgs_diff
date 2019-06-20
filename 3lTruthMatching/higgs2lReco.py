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
from dict_higgs2l import higgsDict
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

    if e.trilep_type==0: continue
    if e.nJets<2: continue
    if e.nJets_MV2c10_70==0: continue
    if len(e.lep_pt)!=3: continue

    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(e.met, 0, e.met_phi, e.met)
    
    lepH = []
    lepB = []

    leps = []

    decay = 0

    for i in range(3):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(e.lep_pt[i], e.lep_eta[i], e.lep_phi[i], e.lep_E[i])
        leps.append(lep)

        if e.lep_parent[i]==25:
            lepH.append(i)
        else:
            lepB.append(i)

    if len(lepH)!=2: continue

    i,j = random.sample(lepH,2)

    k = higgsDict( leps[i], leps[j], leps[lepB[0]], met, 
                   1 )
    eventsFlat.append(k)

    k = higgsDict( leps[i], leps[lepB[0]], leps[j], met,
                   0 )
    eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outDir+'/'+dsid+'Flat.csv', index=False)
