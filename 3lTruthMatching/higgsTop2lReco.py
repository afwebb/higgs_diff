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
from dict_higgsTop2l import higgsTop2lDict
from dict_top3l import topDict
import xgboost as xgb
import pickle
import pandas as pd
#import matplotlib.pyplot as plt

inf = sys.argv[1]
outDir = sys.argv[3]
modelPath = sys.argv[2]
xgbModel = pickle.load(open(modelPath, "rb"))

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

    jet4Vecs = []
    jet4VecsMV2c10 = []
    for i in range(len(nom.jet_pt)):
        jet_pt = nom.jet_pt[i]
        jet_eta = nom.jet_eta[i]
        jet_phi = nom.jet_phi[i]
        jet_E = nom.jet_E[i]
        jet_flav = nom.jet_flavor[i]
        jet_MV2c10 = nom.jet_MV2c10[i]

        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_E)

        jet4Vecs.append(jetVec)
        jet4VecsMV2c10.append(jet_MV2c10)

    combos = []

    if len(jet4Vecs)<2:
        continue

    for i in range(len(jet4Vecs)-1):
        for j in range(i+1, len(jet4Vecs)):
            comb = [i,j]
            t = topDict( jet4Vecs[i], jet4Vecs[j], leps[0], leps[1], leps[2], met,
                         jet4VecsMV2c10[i], jet4VecsMV2c10[j],
                         nom.jet_numTrk[i], nom.jet_numTrk[j])
            combos.append([t, comb])

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)
    bestComb = combos[best][1]

    top1 = jet4Vecs[bestComb[0]]
    top2 = jet4Vecs[bestComb[1]]
    topScore = pred[best]

    if 1 in lepH:
        k = higgsTop2lDict( leps[0], leps[1], leps[2], met, top1, top2, topScore, 1 )
        eventsFlat.append(k)

        k = higgsTop2lDict( leps[0], leps[2], leps[1], met, top1, top2, topScore, 0 )
        eventsFlat.append(k)
    elif 2 in lepH:
        k = higgsTop2lDict( leps[0], leps[2], leps[1], met, top1, top2, topScore, 1 )
        eventsFlat.append(k)

        k = higgsTop2lDict( leps[0], leps[1], leps[2], met, top1, top2, topScore, 0 )
        eventsFlat.append(k)
    else:
        print("lep 0 not from Higgs?")


dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv(outDir+'/'+dsid+'Flat.csv', index=False)
