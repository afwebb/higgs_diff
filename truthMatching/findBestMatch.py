import rootpy.io
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
import sys
import pandas as pd
import pickle
import math
import xgboost as xgb
import numpy as np
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

inf = sys.argv[1]
modelPath = sys.argv[2]
outFile = sys.argv[3]
#njet = sys.argv[3]
f = rootpy.io.root_open(inf)
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

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
nMatch = 0

for e in oldTree:
    current+=1
    if current%10000==0:
        print(current)
        #if current%100000==0:
        #break
 
    #if e.is2LSS0Tau==0 and e.is2LSS1Tau==0: continue
    if e.dilep_type==0: continue
    if abs(e.total_charge)!=2: continue
    #if e.nJets_OR_T<nj: continue
    #if e.higgsDecayMode != 3: continue

    lepPts = [e.lep_Pt_0, e.lep_Pt_1]
    lepEtas = [e.lep_Eta_0, e.lep_Eta_1]
    lepPhis = [e.lep_Phi_0, e.lep_Phi_1]
    lepEs = [e.lep_E_0, e.lep_E_1]
    lepIDs = [e.lep_ID_0, e.lep_ID_1]

    higgCand = LorentzVector()

    lep4Vecs = []
    jet4Vecs = []

    btags = []

    met = LorentzVector()
    met.SetPtEtaPhiE(e.MET_RefFinal_et, 0, e.MET_RefFinal_phi, e.MET_RefFinal_et)

    for i in range(2):
        lepVec = LorentzVector()
        lepVec.SetPtEtaPhiE(lepPts[i], lepEtas[i], lepPhis[i], lepEs[i])
        lep4Vecs.append(lepVec)

    for j in range(len(e.m_jet_pt)):#la[b'selected_jets'][i]:
        jetVec = LorentzVector()
        jetVec.SetPtEtaPhiE(e.m_jet_pt[j], e.m_jet_eta[i], e.m_jet_phi[i], e.m_jet_E[i])
        jet4Vecs.append(jetVec)

        btags.append(e.m_jet_flavor_weight_MV2c10[i])

    combos = []

    for l in range(len(lep4Vecs)):
        for i in range(len(jet4Vecs)-1):
            for j in range(i+1, len(jet4Vecs)):
                comb = [l,i,j]
                
                k = {}
                
                k['dRjj'] = jet4Vecs[i].DeltaR(jet4Vecs[j])
                k['Ptjj'] = (jet4Vecs[i]+jet4Vecs[j]).Pt()
                k['Mjj'] = (jet4Vecs[i]+jet4Vecs[j]).M()
                
                k['dRlj0'] = lep4Vecs[l].DeltaR(jet4Vecs[i])
                k['Ptlj0'] = (lep4Vecs[l]+jet4Vecs[i]).Pt()
                k['Mlj0'] = (lep4Vecs[l]+jet4Vecs[i]).M()
                
                k['dRlj1'] = lep4Vecs[l].DeltaR(jet4Vecs[j])
                k['Ptlj1'] = (lep4Vecs[l]+jet4Vecs[j]).Pt()
                k['Mlj1'] = (lep4Vecs[l]+jet4Vecs[j]).M()
                
                k['MlepMet'] = (lep4Vecs[l]+met).M()
                k['dRlepMet'] = lep4Vecs[l].DeltaR(met)
                
                k['dRj0Met'] = jet4Vecs[i].DeltaR(met)
                k['dRj1Met'] = jet4Vecs[j].DeltaR(met)
                
                k['MhiggsCand'] = (jet4Vecs[i]+jet4Vecs[j]+lep4Vecs[l]+met).M()

                combos.append([k, comb])

    #loop over combinations, score them in the BDT, figure out the best result
    df = pd.DataFrame.from_dict([x[0] for x in combos])
    xgbMat = xgb.DMatrix(df, feature_names=list(df))

    pred = xgbModel.predict(xgbMat)
    best = np.argmax(pred)

    bestScores.append(pred[best])

    #print(pred[best], best) 

    bestComb = combos[best][1]
    lepMatch = bestComb[0]
    jetMatches = bestComb[1:]

    k = {}
    k['higgs_pt'] = e.higgs_pt
    k['comboScore'] = pred[best]

    if lepMatch == 0:
        k['lep_Pt_H'] = e.lep_Pt_0
        k['lep_Eta_H'] = e.lep_Eta_0
        phi_0 = e.lep_Phi_0
        k['lep_E_H'] = e.lep_E_0

        k['lep_Pt_O'] = e.lep_Pt_1
        k['lep_Eta_O'] = e.lep_Eta_1
        k['lep_Phi_O'] = calc_phi(phi_0, e.lep_Phi_1)
        k['lep_E_O'] = e.lep_E_1

    elif lepMatch == 1:
        k['lep_Pt_H'] = e.lep_Pt_1
        k['lep_Eta_H'] = e.lep_Eta_1
        phi_0 = e.lep_Phi_1
        k['lep_E_H'] = e.lep_E_1

        k['lep_Pt_O'] = e.lep_Pt_0
        k['lep_Eta_O'] = e.lep_Eta_0
        k['lep_Phi_O'] = calc_phi(phi_0, e.lep_Phi_0)
        k['lep_E_O'] = e.lep_E_0

    n = 0
    for i in jetMatches:#e.nJets_OR_T):

        k['jet_Pt_h'+str(n)] = e.m_jet_pt[i]
        k['jet_Eta_h'+str(n)] = e.m_jet_eta[i]
        k['jet_E_h'+str(n)] = e.m_jet_E[i]
        k['jet_Phi_h'+str(n)] = calc_phi(phi_0, e.m_jet_phi[i])
        
        n+=1

    btags = np.array(btags)

    btags[jetMatches[0]] = 0
    btags[jetMatches[1]] = 0
    bestBtags = np.argpartition(btags, -2)[-2:]

    n = 0
    for i in bestBtags:#e.nJets_OR_T):                                                                                                         
        
        k['jet_Pt_b'+str(n)] = e.m_jet_pt[i]
        k['jet_Eta_b'+str(n)] = e.m_jet_eta[i]
        k['jet_E_b'+str(n)] = e.m_jet_E[i]
        k['jet_Phi_b'+str(n)] = calc_phi(phi_0, e.m_jet_phi[i])

        n+=1

    k['MET'] = e.MET_RefFinal_et
    k['MET_phi'] = calc_phi(phi_0, e.MET_RefFinal_phi)

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e) 
    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(outFile, index=False)

#plt.figure()
#plt.hist(bestScores)
#plt.savefig('plots/bestScores/'+dsid+'.png')
