import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math

inf = sys.argv[1]
njet = sys.argv[3]
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

    if e.total_leptons!=2: continue
    if abs(e.total_charge)!=2: continue
    if e.nJets_OR_T_MV2c10_70<1: continue
    if e.nJets_OR_T<4: continue

    k = {}
    k['higgs_pt'] = e.higgs_pt

    k['lep_Pt_0'] = e.lep_Pt_0
    k['lep_Eta_0'] = e.lep_Eta_0
    phi_0 = e.lep_Phi_0
    k['lep_E_0'] = e.lep_E_0

    k['lep_Pt_1'] = e.lep_Pt_1
    k['lep_Eta_1'] = e.lep_Eta_1
    k['lep_Phi_1'] = calc_phi(phi_0, e.lep_Phi_1)
    k['lep_E_1'] = e.lep_E_1

    k['MET'] = e.MET_RefFinal_et
    k['MET_phi'] = calc_phi(phi_0, e.MET_RefFinal_phi)

    k['DRlj00'] = e.DRlj00
    k['DRjj01'] = e.DRjj01
    
    if e.nJets_OR_T==int(njet):
        n = 0

        for i in e.selected_jets_T:
            k['jet_Pt_'+str(n)] = e.m_jet_pt[i]
            k['jet_Eta_'+str(n)] = e.m_jet_eta[i]
            k['jet_E_'+str(n)] = e.m_jet_E[i]

            wBtag = e.m_jet_flavor_weight_MV2c10[i]
            if wBtag > 0.94:
                k['jet_MV2c10_'+str(n)] = 4
            elif wBtag > 0.83:
                k['jet_MV2c10_'+str(n)] = 3
            elif wBtag > 0.64:
                k['jet_MV2c10_'+str(n)] = 2
            elif wBtag > 0.11:
                k['jet_MV2c10_'+str(n)] = 1
            else:
                k['jet_MV2c10_'+str(n)] = 0

            k['jet_Phi_'+str(n)] = calc_phi(phi_0, e.m_jet_phi[i])

            n+=1
    else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(sys.argv[2])
