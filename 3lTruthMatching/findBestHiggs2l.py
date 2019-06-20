import uproot
import pandas as pd
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
import sys
import pickle
import math
import xgboost as xgb
import numpy as np
from dict_top3l import topDict
from dict_higgs2l import higgsDict
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

inf = sys.argv[1]
modelPath = sys.argv[2]
topModelPath = sys.argv[3]
outFile = sys.argv[4]
#njet = sys.argv[3]
f = uproot.open(inf)
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print(dsid)
oldTree = f.get('nominal')
    #oldTree.SetBranchStatus("*",0)
    #for br in branch_list:
    #    oldTree.SetBranchStatus(br,1)

events = []
bestScores = []

xgbModel = pickle.load(open(modelPath, "rb"))

topModel = pickle.load(open(topModelPath, "rb"))

la=f['nominal'].lazyarrays(['higgs_pt', 'jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*', 
                            'nJets*', 'total_*', 'dilep*', 'trilep*', 'nJets_MV2c10_70' ])

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
nMatch = 0
totalEvt = len(la[b'met'])

for idx in range(len(la[b'met']) ):
    current+=1
    if current%10000==0:
        print(str(current)+'/'+str(totalEvt))
        #if current%100000==0:
        #break
 
    if la[b'trilep_type'][idx]==0: continue
    if la[b'nJets'][idx]<2: continue
    if la[b'nJets_MV2c10_70'][idx]==0: continue
    if len(la[b'lep_pt'][idx])!=3: continue

    higgCand = LorentzVector()

    lep4Vecs = []
    jet4Vecs = []

    btags = []

    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])

    lepH = []

    for i in range(3):
        lep_pt = la[b'lep_pt'][idx][i]
        lep_eta = la[b'lep_eta'][idx][i]
        lep_phi = la[b'lep_phi'][idx][i]
        lep_E = la[b'lep_E'][idx][i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        lep4Vecs.append(lepVec)

        if la[b'lep_parent'][idx][i]==25:
            lepH.append(i)

    if len(lepH)!=2: continue


    for j in range(len(la[b'jet_pt'][idx])):#la[b'selected_jets'][i]:
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(la[b'jet_pt'][idx][j], la[b'jet_eta'][idx][j], la[b'jet_phi'][idx][j], la[b'jet_E'][idx][j])
        jet4Vecs.append(jetVec)

        btags.append(la[b'jet_MV2c10'][idx][j])

    combos = []
    combosTop = []

    k = higgsDict( lep4Vecs[ 0 ], lep4Vecs[ 1 ], lep4Vecs[ 2 ], met)
    combos.append([k, 1])

    k = higgsDict( lep4Vecs[ 0 ], lep4Vecs[ 2 ], lep4Vecs[ 2 ], met)
    combos.append([k, 2])


    for i in range(len(jet4Vecs)-1):
        for j in range(i+1, len(jet4Vecs)):
            comb = [i,j]
            
            t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], lep4Vecs[2], met, 
                         btags[i], btags[j],
                         la[b'jet_jvt'][idx][i], la[b'jet_jvt'][idx][j],
                         la[b'jet_numTrk'][idx][i], la[b'jet_numTrk'][idx][j])
            
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
    k['higgs_pt'] = la[b'higgs_pt'][idx]
    k['comboScore'] = pred[best]

    if lepMatch == 1:
        k['lep_Pt_0'] = la[b'lep_pt'][idx][0]
        k['lep_Eta_0'] = la[b'lep_eta'][idx][0]
        phi_0 = la[b'lep_phi'][idx][0]
        k['lep_E_0'] = la[b'lep_E'][idx][0]

        k['lep_Pt_1'] = la[b'lep_pt'][idx][1]
        k['lep_Eta_1'] = la[b'lep_eta'][idx][1]
        k['lep_Phi_1'] = calc_phi(phi_0, la[b'lep_phi'][idx][1])
        k['lep_E_1'] = la[b'lep_E'][idx][1]

        k['lep_Pt_2'] = la[b'lep_pt'][idx][2]
        k['lep_Eta_2'] = la[b'lep_eta'][idx][2]
        k['lep_Phi_2'] = calc_phi(phi_0, la[b'lep_phi'][idx][2])
        k['lep_E_2'] = la[b'lep_E'][idx][2]

    elif lepMatch == 2:
        k['lep_Pt_0'] = la[b'lep_pt'][idx][0]
        k['lep_Eta_0'] = la[b'lep_eta'][idx][0]
        phi_0 = la[b'lep_phi'][idx][0]
        k['lep_E_0'] = la[b'lep_E'][idx][0]

        k['lep_Pt_1'] = la[b'lep_pt'][idx][2]
        k['lep_Eta_1'] = la[b'lep_eta'][idx][2]
        k['lep_Phi_1'] = calc_phi(phi_0, la[b'lep_phi'][idx][2])
        k['lep_E_1'] = la[b'lep_E'][idx][2]

        k['lep_Pt_2'] = la[b'lep_pt'][idx][1]
        k['lep_Eta_2'] = la[b'lep_eta'][idx][1]
        k['lep_Phi_2'] = calc_phi(phi_0, la[b'lep_phi'][idx][1])
        k['lep_E_2'] = la[b'lep_E'][idx][1]

    n = 0
    for i in topMatches:#bestBtags:#la[b'nJets_OR_T):      
        k['top_Pt_'+str(n)] = la[b'jet_pt'][idx][i]
        k['top_Eta_'+str(n)] = la[b'jet_eta'][idx][i]
        k['top_E_'+str(n)] = la[b'jet_E'][idx][i]
        k['top_Phi_'+str(n)] = calc_phi(phi_0, la[b'jet_phi'][idx][i])
        k['top_MV2c10_'+str(n)] = la[b'jet_MV2c10'][idx][i]

        n+=1

    k['MET'] = la[b'met'][idx]
    k['MET_phi'] = calc_phi(phi_0, la[b'met_phi'][idx])

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e) 
    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(outFile, index=False)

#plt.figure()
#plt.hist(bestScores)
#plt.savefig('plots/bestScores/'+dsid+'.png')
