#import ROOT
import numpy as np
#import rootpy.io
import uproot
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
f=uproot.open(inputFile)
dsid = inputFile.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
nom = f.get('nominal')

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

la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*', 'trilep_type', 'nJets', 'nJets_MV2c10_70', 'jet_numTrk'])

print(la[b'jet_numTrk'][4])

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
    #if current%200==0:
    #    break

    if la[b'trilep_type'][idx]==0: continue
    if la[b'nJets'][idx]<2: continue
    if la[b'nJets_MV2c10_70'][idx]==0: continue
    if len(la[b'lep_pt'][idx])!=3: continue
          
    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])
    
    if len(la[b'lep_pt'][idx])!=3: continue

    lepH = []
    lepB = []

    leps = []

    decay = 0

    for i in range(3):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(la[b'lep_pt'][idx][i], la[b'lep_eta'][idx][i], la[b'lep_phi'][idx][i], la[b'lep_E'][idx][i])
        leps.append(lep)

        if la[b'lep_parent'][idx][i]==25:
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
        
    #if len(lepH)!=1 or len(higgsJets)!=2 or len(topJets)!=2: continue

    combosTop = []
    combos1l = []

    for l in range(len(leps)):
        for i in range(len(jets)-1):
            for j in range(i+1, len(jets)):
                comb = [l,i,j]
                t = topDict( jets[i], jets[j], leps[0], leps[1], leps[2], met, la[b'jet_MV2c10'][idx][i], la[b'jet_MV2c10'][idx][j],
                             la[b'jet_jvt'][idx][i], la[b'jet_jvt'][idx][j], la[b'jet_numTrk'][idx][i], la[b'jet_numTrk'][idx][j])
                #t['nJets'] = la[b'nJets'][idx]
                #t['nJets_MV2c10_70'] = la[b'nJets_MV2c10_70'][idx]
                combosTop.append([t, comb])

                if l==0:
                    k = higgs1lDict( jets[i], jets[j], leps[l], met, la[b'jet_MV2c10'][idx][i], la[b'jet_MV2c10'][idx][j], leps[1],
                                   la[b'jet_jvt'][idx][i], la[b'jet_jvt'][idx][j],
                                   la[b'jet_numTrk'][idx][i], la[b'jet_numTrk'][idx][j])
                else:
                    k = higgs1lDict( jets[i], jets[j], leps[l], met, la[b'jet_MV2c10'][idx][i], la[b'jet_MV2c10'][idx][j], leps[0],
                                   la[b'jet_jvt'][idx][i], la[b'jet_jvt'][idx][j],
                                   la[b'jet_numTrk'][idx][i], la[b'jet_numTrk'][idx][j])

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
    k['nJets'] = la[b'nJets'][idx]
    k['nJets_MV2c10_70'] = la[b'nJets_MV2c10_70'][idx]
    k['higgs2l_score'] = pred2l[best2l]
    k['higgs1l_score'] = pred1l[best1l]
    eventsFlat.append(k)

import pandas as pd

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('decay3lXGBTopFiles/'+dsid+'Flat.csv', index=False)
#dfFlat.to_csv('test'+dsid+'Flat.csv', index=False)
