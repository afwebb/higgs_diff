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
from dict_higgsTop1l import higgsTop1lDict
from dict_top3l import topDict
import pandas as pd
import xgboost as xgb
import pickle
#import matplotlib.pyplot as plt

inf = sys.argv[1]
outDir = sys.argv[3]
modelPath = sys.argv[2]
xgbModel = pickle.load(open(modelPath, "rb"))

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
    leps = []

    for i in range(3):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(nom.lep_pt[i], nom.lep_eta[i], nom.lep_phi[i], nom.lep_E[i])
        leps.append(lep)
        if nom.lep_parent[i]==25:
            lepH.append(lep)
        else:
            lepB.append(lep)

    jets = []
    jetsMV2c10 = []
    higgsJets = []
    badJets = []
    for i in range(len(nom.jet_pt)):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(nom.jet_pt[i], nom.jet_eta[i], nom.jet_phi[i], nom.jet_E[i])
        jets.append(jet)
        
        jetsMV2c10.append(nom.jet_MV2c10[i])

        if nom.jet_parent[i]==25:
            higgsJets.append(i)
        else:
            badJets.append(i)
        
    if len(lepH)!=1 or len(higgsJets)!=2: continue

    combos = []
    
    for i in range(len(jets)-1):
        for j in range(i+1, len(jets)):
            comb = [i,j]
            t = topDict( jets[i], jets[j], leps[0], leps[1], leps[2], met,
                         jetsMV2c10[i], jetsMV2c10[j],
                         nom.jet_numTrk[i], nom.jet_numTrk[j])
            combos.append([t, comb])
            
    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)
    bestComb = combos[best][1]

    top1 = jets[bestComb[0]]
    top2 = jets[bestComb[1]]
    topScore = pred[best]

    k = higgsTop1lDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepH[0], met, 
                     nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[ higgsJets[1] ], lepB[0], lepB[1],
                     nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[ higgsJets[1] ], 
                     top1, top2, topScore,
                     1 )
    eventsFlat.append(k)

    for l in range(2):
        k = higgsTop1lDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepB[0], met, 
                         nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[ higgsJets[1] ], lepH[0], lepB[1],
                         nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[ higgsJets[1] ], 
                         top1, top2, topScore,
                         0 )
        eventsFlat.append(k)

    for l in range(2):
        if len(badJets)<2: break
        i,j = random.sample(badJets,2)
        k = higgsTop1lDict( jets[i], jets[j], lepH[0], met, 
                         nom.jet_MV2c10[i], nom.jet_MV2c10[j], lepB[0], lepB[1],
                         nom.jet_numTrk[i], nom.jet_numTrk[j],
                         top1, top2, topScore,
                         0 )
        eventsFlat.append(k)

        k = higgsTop1lDict( jets[i], jets[ higgsJets[1] ], lepH[0], met, 
                         nom.jet_MV2c10[i], nom.jet_MV2c10[ higgsJets[1]], lepB[0], lepB[1], 
                         nom.jet_numTrk[i], nom.jet_numTrk[ higgsJets[1]],
                         top1, top2, topScore,
                         0 )
        eventsFlat.append(k)
        
        k = higgsTop1lDict( jets[ higgsJets[0] ], jets[j], lepH[0], met, 
                         nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[j], lepB[0], lepB[1],
                         nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[j],
                         top1, top2, topScore,
                         0 )
        eventsFlat.append(k)

    for l in range(min([6, len(badJets)])):
        if len(badJets)<2: break
        i,j = random.sample(badJets,2)
        k = higgsTop1lDict( jets[i], jets[j], lepB[0], met, 
                         nom.jet_MV2c10[i], nom.jet_MV2c10[j], lepH[0], lepB[1],
                         nom.jet_numTrk[i], nom.jet_numTrk[j],
                         top1, top2, topScore,
                         0 )
        eventsFlat.append(k)

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outDir+'/'+dsid+'Flat.csv', index=False)
