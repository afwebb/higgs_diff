#import ROOT
import numpy as np
#import rootpy.io
import ROOT
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
import xgboost as xgb
import pickle
import pandas as pd
from dict_3lDecay import decayDict
from dict_top3l import topDict
from dict_higgs2l import higgs2lDict
from dict_higgs1l import higgs1lDict

#import matplotlib.pyplot as plt

inputFile = sys.argv[1]
#outputFile = sys.argv[2]
topModelPath = sys.argv[2]
twoLepModelPath = sys.argv[3]
oneLepModelPath = sys.argv[4]

#f = rootpy.io.root_open(inf)
f=ROOT.TFile.Open(inputFile, "READ")
dsid = inputFile.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
nom = f.Get('nominal')

topModel = pickle.load(open(topModelPath, "rb"))
twoLepModel = pickle.load(open(twoLepModelPath, "rb"))
oneLepModel = pickle.load(open(oneLepModelPath, "rb"))

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
    
    if len(nom.lep_pt)!=3: continue

    lepH = []
    lepB = []

    leps = []

    decay = 0

    for i in range(3):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(nom.lep_pt[i], nom.lep_eta[i], nom.lep_phi[i], nom.lep_E[i])
        leps.append(lep)

        if nom.lep_parent[i]==25:
            lepH.append(lep)
        else:
            lepB.append(lep)

    if len(lepH)==2:
        decay = 0
    elif len(lepH)==1:
        decay = 1
    else:
        continue

    jets = []
    higgsJets = []
    topJets = []
    badJets = []
    for i in range(len(nom.jet_pt)):
        jet = LorentzVector()
        jet.SetPtEtaPhiE(nom.jet_pt[i], nom.jet_eta[i], nom.jet_phi[i], nom.jet_E[i])
        jets.append(jet)
        
        if nom.jet_parent[i]==25:
            higgsJets.append(i)
        elif abs(nom.jet_parent[i])==6:
            topJets.append(i)
        else:
            badJets.append(i)
        
    #if len(lepH)!=1 or len(higgsJets)!=2 or len(topJets)!=2: continue

    combosTop = []
    combos1l = []

    for l in range(len(leps)):
        for i in range(len(jets)-1):
            for j in range(i+1, len(jets)):
                comb = [l,i,j]
                t = topDict( jets[i], jets[j], leps[0], leps[1], leps[2], met, nom.jet_MV2c10[i], nom.jet_MV2c10[j],
                             nom.jet_jvt[i], nom.jet_jvt[j], nom.jet_numTrk[i], nom.jet_numTrk[j])
                #t['nJets'] = nom.nJets
                #t['nJets_MV2c10_70'] = nom.nJets_MV2c10_70
                combosTop.append([t, comb])

                if l==1:
                    k = higgs1lDict( jets[i], jets[j], leps[l], met, nom.jet_MV2c10[i], nom.jet_MV2c10[j], leps[0], leps[2],
                                   nom.jet_jvt[i], nom.jet_jvt[j],
                                   nom.jet_numTrk[i], nom.jet_numTrk[j])
                else:
                    k = higgs1lDict( jets[i], jets[j], leps[l], met, nom.jet_MV2c10[i], nom.jet_MV2c10[j], leps[0], leps[2],
                                   nom.jet_jvt[i], nom.jet_jvt[j],
                                   nom.jet_numTrk[i], nom.jet_numTrk[j])

                combos1l.append([k, comb])


    topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
    topMat = xgb.DMatrix(topDF, feature_names=list(topDF))

    topPred = topModel.predict(topMat)
    topBest = np.argmax(topPred)

    bestTopComb = combosTop[topBest][1]
    topMatches = bestTopComb[1:]

    combos2l = []

    possCombs = [[0,1,2],[0,2,1]]
    for comb in possCombs:
        k = higgs2lDict( leps[ comb[0] ], leps[ comb[1] ], leps[ comb[2] ], met)
        combos2l.append([k, [comb[0], comb[1]] ])

    #Run 2l XGB, find best match
    df2l = pd.DataFrame.from_dict([x[0] for x in combos2l])
    xgbMat2l = xgb.DMatrix(df2l, feature_names=list(df2l))

    pred2l = twoLepModel.predict(xgbMat2l)
    best2l = np.argmax(pred2l)

    #Run 1l XGB, find best match
    df1l = pd.DataFrame.from_dict([x[0] for x in combos1l])
    xgbMat1l = xgb.DMatrix(df1l, feature_names=list(df1l))

    pred1l = oneLepModel.predict(xgbMat1l)
    best1l = np.argmax(pred1l)

    k = decayDict( leps[0], leps[1], leps[2], met, jets[ topMatches[0] ], jets[ topMatches[1] ], decay )
    k['nJets'] = nom.nJets
    k['nJets_MV2c10_70'] = nom.nJets_MV2c10_70
    k['higgs2l_score'] = pred2l[best2l]
    k['higgs1l_score'] = pred1l[best1l]
    eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('decay3lXGBTopFiles/'+dsid+'Flat.csv', index=False)
#dfFlat.to_csv('test'+dsid+'Flat.csv', index=False)
