import ROOT
import pandas as pd
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
import sys
import pickle
import math
import xgboost as xgb
import numpy as np
from dict_top3l import topDict
from dict_higgs2l import higgs2lDict
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

inf = sys.argv[1]
modelPath = sys.argv[2]
topModelPath = sys.argv[3]
outFile = sys.argv[4]
#njet = sys.argv[3]
f = ROOT.TFile.Open(inf)
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
nom = f.Get('nominal')

events = []
bestScores = []

xgbModel = pickle.load(open(modelPath, "rb"))

topModel = pickle.load(open(topModelPath, "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
nMatch = 0

nEntries = nom.GetEntries()
for idx in range(nEntries):
    if idx%10000==0:
        print(str(idx)+'/'+str(nEntries))
        #if current%100000==0:
        #break
    nom.GetEntry(idx)
 
    if nom.trilep_type==0: continue
    if nom.nJets<2: continue
    if nom.nJets_MV2c10_70==0: continue
    if len(nom.lep_pt)!=3: continue
    if nom.lep_pt[0]<10000: continue
    if nom.lep_pt[1]<20000: continue
    if nom.lep_pt[2]<20000: continue

    higgCand = LorentzVector()

    lep4Vecs = []
    jet4Vecs = []

    btags = []

    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met, 0, nom.met_phi, nom.met)

    lepH = []

    for i in range(3):
        lep_pt = nom.lep_pt[i]
        lep_eta = nom.lep_eta[i]
        lep_phi = nom.lep_phi[i]
        lep_E = nom.lep_E[i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        lep4Vecs.append(lepVec)

        if nom.lep_parent[i]==25:
            lepH.append(i)

    if len(lepH)!=2: continue


    for j in range(len(nom.jet_pt)):#nom.selected_jets'][i]:
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(nom.jet_pt[j], nom.jet_eta[j], nom.jet_phi[j], nom.jet_E[j])
        jet4Vecs.append(jetVec)

        btags.append(nom.jet_MV2c10[j])

    combos = []
    combosTop = []

    k = higgs2lDict( lep4Vecs[ 0 ], lep4Vecs[ 1 ], lep4Vecs[ 2 ], met)
    combos.append([k, 1])

    k = higgs2lDict( lep4Vecs[ 0 ], lep4Vecs[ 2 ], lep4Vecs[ 2 ], met)
    combos.append([k, 2])


    for i in range(len(jet4Vecs)-1):
        for j in range(i+1, len(jet4Vecs)):
            comb = [i,j]
            
            t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met, 
                         btags[i], btags[j],
                         nom.jet_jvt[i], nom.jet_jvt[j],
                         nom.jet_numTrk[i], nom.jet_numTrk[j])
            
            combosTop.append([t, comb])

    #loop over combinations, score them in the BDT, figure out the best result
    topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
    topMat = xgb.DMatrix(topDF, feature_names=list(topDF))

    topPred = topModel.predict(topMat)
    topBest = np.argmax(topPred)

    bestTopComb = combosTop[topBest][1]
    topMatches = bestTopComb

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    lepMatch = bestComb

    k = {}
    k['higgs_pt'] = nom.higgs_pt
    k['comboScore'] = pred[best]

    if lepMatch == 1:
        k['lep_Pt_0'] = nom.lep_pt[0]
        k['lep_Eta_0'] = nom.lep_eta[0]
        phi_0 = nom.lep_phi[0]
        k['lep_E_0'] = nom.lep_E[0]

        k['lep_Pt_1'] = nom.lep_pt[1]
        k['lep_Eta_1'] = nom.lep_eta[1]
        k['lep_Phi_1'] = calc_phi(phi_0, nom.lep_phi[1])
        k['lep_E_1'] = nom.lep_E[1]

        k['lep_Pt_2'] = nom.lep_pt[2]
        k['lep_Eta_2'] = nom.lep_eta[2]
        k['lep_Phi_2'] = calc_phi(phi_0, nom.lep_phi[2])
        k['lep_E_2'] = nom.lep_E[2]

    elif lepMatch == 2:
        k['lep_Pt_0'] = nom.lep_pt[0]
        k['lep_Eta_0'] = nom.lep_eta[0]
        phi_0 = nom.lep_phi[0]
        k['lep_E_0'] = nom.lep_E[0]

        k['lep_Pt_1'] = nom.lep_pt[2]
        k['lep_Eta_1'] = nom.lep_eta[2]
        k['lep_Phi_1'] = calc_phi(phi_0, nom.lep_phi[2])
        k['lep_E_1'] = nom.lep_E[2]

        k['lep_Pt_2'] = nom.lep_pt[1]
        k['lep_Eta_2'] = nom.lep_eta[1]
        k['lep_Phi_2'] = calc_phi(phi_0, nom.lep_phi[1])
        k['lep_E_2'] = nom.lep_E[1]

    n = 0
    for i in topMatches:#bestBtags:#nom.nJets_OR_T):      
        k['top_Pt_'+str(n)] = nom.jet_pt[i]
        k['top_Eta_'+str(n)] = nom.jet_eta[i]
        k['top_E_'+str(n)] = nom.jet_E[i]
        k['top_Phi_'+str(n)] = calc_phi(phi_0, nom.jet_phi[i])
        k['top_MV2c10_'+str(n)] = nom.jet_MV2c10[i]

        n+=1

    k['MET'] = nom.met
    k['MET_phi'] = calc_phi(phi_0, nom.met_phi)

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e) 
    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(outFile, index=False)

#plt.figure()
#plt.hist(bestScores)
#plt.savefig('plots/bestScores/'+dsid+'.png')
