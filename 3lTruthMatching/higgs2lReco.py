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
from dict_higgs2l import higgs2lDict
import pandas as pd
#import matplotlib.pyplot as plt

inf = sys.argv[1]
outDir = sys.argv[2]

f = ROOT.TFile(inf, "READ")
nom=f.Get('nominal')
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)

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

    leps = []

    decay = 0

    for i in range(3):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(nom.lep_pt[i], nom.lep_eta[i], nom.lep_phi[i], nom.lep_E[i])
        leps.append(lep)

        if nom.lep_parent[i]==25:
            lepH.append(i)
        else:
            lepB.append(i)

    if len(lepH)!=2: continue

    i,j = random.sample(lepH,2)

    if 1 in lepH:
        k = higgs2lDict( leps[0], leps[1], leps[2], met, 1 )
        eventsFlat.append(k)
        
        k = higgs2lDict( leps[0], leps[2], leps[1], met, 0 )
        eventsFlat.append(k)
    elif 2 in lepH:
        k = higgs2lDict( leps[0], leps[2], leps[1], met, 1 )
        eventsFlat.append(k)
        
        k = higgs2lDict( leps[0], leps[1], leps[2], met, 0 )
        eventsFlat.append(k)
    else:
        print("lep 0 not from Higgs?")
    #k = higgs2lDict( leps[lepB[0]], leps[i], leps[j], met, 
    #               0 )
    #eventsFlat.append(k)

#import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outDir+'/'+dsid+'Flat.csv', index=False)