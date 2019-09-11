import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math

inf = sys.argv[1]
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

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

current = 0
for e in oldTree:
    current+=1
    if current%10000==0:
        print(current)

    #if e.total_leptons!=2: continue
    #if abs(e.total_charge)!=2: continue
    #if e.nJets_OR_T_MV2c10_70<1: continue
    #if e.nJets_OR_T<4: continue

    if e.is2LSS0Tau==0 and e.is2LSS1Tau==0: continue
    if e.dilep_type==0: continue
    if abs(e.total_charge)!=2: continue
    if e.nJets_OR_T<4: continue
    if e.nJets_OR_T_MV2c10_70<1: continue

    k = {}

    if '34587' in dsid or '34567' in dsid or '34634' in dsid:
        k['signal'] = 1
    else:
        k['signal'] = 0

    k['dilep_type'] = e.dilep_type
    k['lep_Pt_0'] = e.lep_Pt_0
    k['lep_Eta_0'] = e.lep_Eta_0
    k['lep_Phi_0'] = e.lep_Phi_0
    k['lep_ID_0'] = e.lep_ID_0
    k['lep_Pt_1'] = e.lep_Pt_1
    k['lep_Eta_1'] = e.lep_Eta_1
    k['lep_Phi_1'] = e.lep_Phi_1
    k['lep_ID_1'] = e.lep_ID_1
    k['Mll01'] = e.Mll01
    k['DRll01'] = e.DRll01
    k['Ptll01'] = e.Ptll01
    k['lead_jetPt'] = e.lead_jetPt 
    k['lead_jetEta'] = e.lead_jetEta
    k['lead_jetPhi'] = e.lead_jetPhi 
    k['sublead_jetPt'] = e.sublead_jetPt 
    k['sublead_jetEta'] = e.sublead_jetEta
    k['sublead_jetPhi'] = e.sublead_jetPhi
    k['HT'] = e.HT 
    k['HT_lep'] = e.HT_lep
    k['nJets_OR_T'] = e.nJets_OR_T
    k['nJets_OR_T_MV2c10_70'] = e.nJets_OR_T_MV2c10_70 
    k['MET_RefFinal_et'] = e.MET_RefFinal_et
    k['MET_RefFinal_phi'] = e.MET_RefFinal_phi
    k['DRlj00'] = e.DRlj00 
    #k['DRlj10'] = e.DRlj10
    k['DRjj01'] = e.DRjj01

    k['bin_score'] = e.xgb_bin_score_2l
    k['pt_score'] = e.xgb_pt_score_2l
    k['higgsScore'] = e.higgsScore
    k['topScore'] = e.topScore

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)                                                                       
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)

    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(sys.argv[2], index=False)
