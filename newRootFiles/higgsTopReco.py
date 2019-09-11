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
import xgboost as xgb
import pickle
from dict_top import topDict
#import matplotlib.pyplot as plt

inf = sys.argv[1]
topModelPath = sys.argv[2]
#outputFile = sys.argv[2]

f = ROOT.TFile.Open(inf, "READ")
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
nom = f.Get('nominal')

topModel = pickle.load(open(topModelPath, "rb"))

events = []

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

#la=f['nominal'].lazyarrays(['jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*', 
#                            'nJets*', 'total_*', 'dilep*', 'nJets_MV2c10_70' ])

current = 0
nMatch = 0
higgVecs = []

goodMatches = 0
badMatches = 0

fourVecDicts = []
eventsFlat = []
current = 0
#totalEvt = len(la[b'met']) 

nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))
        #if current%100000==0:
        #break
    nom.GetEntry(idx)

    if nom.total_leptons!=2: continue
    if nom.total_charge==0: continue
    if nom.dilep_type<1: continue
    if nom.nJets<4: continue
    if nom.nJets_MV2c10_70<1: continue
    if nom.lep_pt[0]<20000: continue
    if nom.lep_pt[1]<20000: continue

    fourVecs = {}
    
    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met, 0, nom.met_phi, nom.met)
    
    if len(nom.lep_pt)!=2: continue

    lepH = []
    lepB = []
    lep4Vecs = []

    for i in range(2):
        lep = LorentzVector()
        lep.SetPtEtaPhiE(nom.lep_pt[i], nom.lep_eta[i], nom.lep_phi[i], nom.lep_E[i])
        lep4Vecs.append(lep)
        if nom.lep_parent[i]==25:
            lepH.append(lep)
        else:
            lepB.append(lep)

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
        
    if len(lepH)!=1 or len(higgsJets)!=2: continue # or len(topJets)!=2: continue

    combosTop = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jets)-1):
                for j in range(i+1, len(jets)):
                    comb = [l,i,j]
                    t = topDict( jets[i], jets[j], lep4Vecs[0], lep4Vecs[1], met, 
                                 nom.jet_MV2c10[i], nom.jet_MV2c10[j],
                                 nom.jet_jvt[i], nom.jet_jvt[j],
                                 nom.jet_numTrk[i], nom.jet_numTrk[j]
                             )
                    
                    combosTop.append([t, comb])

    topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
    topMat = xgb.DMatrix(topDF, feature_names=list(topDF))

    topPred = topModel.predict(topMat)
    topBest = np.argmax(topPred)

    bestTopComb = combosTop[topBest][1]
    topMatches = bestTopComb[1:]

    k = higgsTopDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepH[0], met, 
                      nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[ higgsJets[1] ], 
                      jets[ topMatches[0] ], jets[ topMatches[1] ], lepB[0], 
                      nom.jet_jvt[ higgsJets[0] ], nom.jet_jvt[ higgsJets[1] ],
                      nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[ higgsJets[1] ],
                      1 )
    k['topScore'] = topPred[topBest]
    eventsFlat.append(k)

    for l in range(2):
        k = higgsTopDict( jets[ higgsJets[0] ], jets[ higgsJets[1] ], lepB[0], met, 
                          nom.jet_MV2c10[ higgsJets[0] ], nom.jet_MV2c10[ higgsJets[1] ], 
                          jets[ topMatches[0] ], jets[ topMatches[1] ], lepH[0], 
                          nom.jet_jvt[ higgsJets[0] ], nom.jet_jvt[ higgsJets[1] ],
                          nom.jet_numTrk[ higgsJets[0] ], nom.jet_numTrk[ higgsJets[1] ],
                          0 )
        k['topScore'] = topPred[topBest]
        eventsFlat.append(k)

    if len(badJets) > 2:
        for l in range(2):
            i,j = random.sample(badJets,2)
            k = higgsTopDict( jets[i], jets[j], lepH[0], met, 
                              nom.jet_MV2c10[i], nom.jet_MV2c10[j], 
                              jets[ topMatches[0] ], jets[ topMatches[1] ], lepB[0], 
                              nom.jet_jvt[i], nom.jet_jvt[j],
                              nom.jet_numTrk[i], nom.jet_numTrk[j],
                              0 )
            k['topScore'] = topPred[topBest]
            eventsFlat.append(k)

        for l in range(min([8, len(badJets)])):
            i,j = random.sample(badJets,2)
            k = higgsTopDict( jets[i], jets[j], lepB[0], met, 
                              nom.jet_MV2c10[i], nom.jet_MV2c10[j], 
                              jets[ topMatches[0] ], jets[ topMatches[1] ], lepH[0], 
                              nom.jet_jvt[i], nom.jet_jvt[j],
                              nom.jet_numTrk[i], nom.jet_numTrk[j],
                              0 )
            k['topScore'] = topPred[topBest]
            eventsFlat.append(k)

dfFlat = pd.DataFrame.from_dict(eventsFlat)

from sklearn.utils import shuffle
dfFlat = shuffle(dfFlat)

dfFlat.to_csv('higgsTopLepCut/'+dsid+'Flat.csv', index=False)
