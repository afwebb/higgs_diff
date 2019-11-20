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
outFile = sys.argv[2]
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

    k = {}

    k['higgs_pt'] = e.higgs_pt

    hLep = -1
    tLep = -1
    hJets = []
    tJets = []

    for i in range(len(e.lep_pt)):
        if e.lep_parent[i]==25:
            hLep = i
        elif e.lep_parent[i]==6:
            tLep = i

    if hLep==-1 or tLep==-1:
        continue

    for i in range(len(e.jet_pt)):
        if e.jet_parent[i]==25:
            hJets.append(i)
        elif e.jet_parent[i]==6 and e.jet_flavor[i]==5:
            tJets.append(i)

    if len(hJets)!=2 or len(tJets)!=2:
        continue

    if hLep == 0:
        k['lep_Pt_H'] = e.lep_pt[0]
        k['lep_Eta_H'] = e.lep_eta[0]
        phi_0 = e.lep_phi[0]
        k['lep_E_H'] = e.lep_E[0]

        k['lep_Pt_O'] = e.lep_pt[1]
        k['lep_Eta_O'] = e.lep_eta[1]
        k['lep_Phi_O'] = calc_phi(phi_0, e.lep_phi[1])
        k['lep_E_O'] = e.lep_E[1]

    else:
        k['lep_Pt_H'] = e.lep_pt[1]
        k['lep_Eta_H'] = e.lep_eta[1]
        phi_0 = e.lep_phi[1]
        k['lep_E_H'] = e.lep_E[1]

        k['lep_Pt_O'] = e.lep_pt[0]
        k['lep_Eta_O'] = e.lep_eta[0]
        k['lep_Phi_O'] = calc_phi(phi_0, e.lep_phi[0])
        k['lep_E_O'] = e.lep_E[0]

    n = 0
    for i in hJets:#e.nJets_OR_T):

        k['jet_Pt_h'+str(n)] = e.jet_pt[i]
        k['jet_Eta_h'+str(n)] = e.jet_eta[i]
        k['jet_E_h'+str(n)] = e.jet_E[i]
        k['jet_Phi_h'+str(n)] = calc_phi(phi_0, e.jet_phi[i])
        k['jet_MV2c10_h'+str(n)] = e.jet_MV2c10[i]
        
        n+=1

    n = 0
    for i in tJets:
        k['top_Pt_'+str(n)] = e.jet_pt[i]
        k['top_Eta_'+str(n)] = e.jet_eta[i]
        k['top_E_'+str(n)] = e.jet_E[i]
        k['top_Phi_'+str(n)] = calc_phi(phi_0, e.jet_phi[i])
        k['top_MV2c10_'+str(n)] = e.jet_MV2c10[i]

        n+=1

    k['MET'] = e.met
    k['MET_phi'] = calc_phi(phi_0, e.met_phi)

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e) 
    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(outFile, index=False)

#plt.figure()
#plt.hist(bestScores)
#plt.savefig('plots/bestScores/'+dsid+'.png')
