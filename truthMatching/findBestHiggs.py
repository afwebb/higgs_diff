import uproot
import pandas as pd
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
import sys
import pickle
import math
import xgboost as xgb
import numpy as np
from dict_top import topDict
from dict_higgs import higgsDict
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

la=f['nominal'].lazyarrays(['higgs_pt', 'jet_*', 'lep_*', 'met', 'met_phi', 'truth_jet_*', 'track_jet_*'])

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
nMatch = 0

for idx in range(len(la[b'met']) ):
    current+=1
    if current%10000==0:
        print(current)
        #if current%100000==0:
        #break
 
    #if e.is2LSS0Tau==0 and e.is2LSS1Tau==0: continue
    #if e.total_leptons != 2: continue
    #if e.dilep_type < 1: continue
    #if e.total_charge == 0: continue
    #if e.nJets_OR_T_MV2c10_70 < 1: continue
    #if e.nJets_OR_T < 4: continue
    #if e.higgsDecayMode != 3: continue

    #lepPts = [e.lep_Pt_0, e.lep_Pt_1]
    #lepEtas = [e.lep_Eta_0, e.lep_Eta_1]
    #lepPhis = [e.lep_Phi_0, e.lep_Phi_1]
    #lepEs = [e.lep_E_0, e.lep_E_1]
    #lepIDs = [e.lep_ID_0, e.lep_ID_1]

    higgCand = LorentzVector()

    lep4Vecs = []
    jet4Vecs = []

    btags = []

    met = LorentzVector()
    met.SetPtEtaPhiE(la[b'met'][idx], 0, la[b'met_phi'][idx], la[b'met'][idx])

    for i in range(2):
        lep_pt = la[b'lep_pt'][idx][i]
        lep_eta = la[b'lep_eta'][idx][i]
        lep_phi = la[b'lep_phi'][idx][i]
        lep_E = la[b'lep_E'][idx][i]

        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lep_pt, lep_eta, lep_phi, lep_E)
        lep4Vecs.append(lepVec)

    for j in range(len(la[b'jet_pt'][idx])):#la[b'selected_jets'][i]:
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(la[b'jet_pt'][idx][j], la[b'jet_eta'][idx][j], la[b'jet_phi'][idx][j], la[b'jet_E'][idx][j])
        jet4Vecs.append(jetVec)

        btags.append(la[b'jet_MV2c10'][idx][j])

    combos = []
    combosTop = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]
                
                if l==0:
                    k = higgsDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, btags[i], btags[j], lep4Vecs[1] )
                else:
                    k = higgsDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[l], met, btags[i], btags[j], lep4Vecs[0] )
                
                combos.append([k, comb])

                #k = flatDict( lep4Vecs[l], jet4Vecs[i], jet4Vecs[j], met, btags[i], btags[j] )
                t = topDict( jet4Vecs[i], jet4Vecs[j], lep4Vecs[0], lep4Vecs[1], met, btags[i], btags[j] )

                combosTop.append([t, comb])

    #loop over combinations, score them in the BDT, figure out the best result
    topDF = pd.DataFrame.from_dict([x[0] for x in combosTop])
    topMat = xgb.DMatrix(topDF, feature_names=list(topDF))

    topPred = topModel.predict(topMat)
    topBest = np.argmax(topPred)

    bestTopComb = combosTop[topBest][1]
    topMatches = bestTopComb[1:]

    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    bestComb = combos[best][1]
    lepMatch = bestComb[0]
    jetMatches = bestComb[1:]

    k = {}
    k['higgs_pt'] = la[b'higgs_pt'][idx]
    k['comboScore'] = pred[best]

    if lepMatch == 0:
        k['lep_Pt_H'] = la[b'lep_pt'][idx][0]
        k['lep_Eta_H'] = la[b'lep_eta'][idx][0]
        phi_0 = la[b'lep_phi'][idx][0]
        k['lep_E_H'] = la[b'lep_E'][idx][0]

        k['lep_Pt_O'] = la[b'lep_pt'][idx][1]
        k['lep_Eta_O'] = la[b'lep_eta'][idx][1]
        k['lep_Phi_O'] = calc_phi(phi_0, la[b'lep_phi'][idx][1])
        k['lep_E_O'] = la[b'lep_E'][idx][1]

    elif lepMatch == 1:
        k['lep_Pt_H'] = la[b'lep_pt'][idx][1]
        k['lep_Eta_H'] = la[b'lep_eta'][idx][1]
        phi_0 = la[b'lep_phi'][idx][1]
        k['lep_E_H'] = la[b'lep_E'][idx][1]

        k['lep_Pt_O'] = la[b'lep_pt'][idx][0]
        k['lep_Eta_O'] = la[b'lep_eta'][idx][0]
        k['lep_Phi_O'] = calc_phi(phi_0, la[b'lep_phi'][idx][0])
        k['lep_E_O'] = la[b'lep_E'][idx][0]

    n = 0
    for i in jetMatches:#la[b'nJets_OR_T):

        k['jet_Pt_h'+str(n)] = la[b'jet_pt'][idx][i]
        k['jet_Eta_h'+str(n)] = la[b'jet_eta'][idx][i]
        k['jet_E_h'+str(n)] = la[b'jet_E'][idx][i]
        k['jet_Phi_h'+str(n)] = calc_phi(phi_0, la[b'jet_phi'][idx][i])
        k['jet_MV2c10_h'+str(n)] = la[b'jet_MV2c10'][idx][i]
        
        n+=1

    btags = np.array(btags)

    btags[jetMatches[0]] = 0
    btags[jetMatches[1]] = 0
    bestBtags = np.argpartition(btags, -2)[-2:]

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
