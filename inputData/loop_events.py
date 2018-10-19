import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math

inf = sys.argv[1]
f = rootpy.io.root_open(inf)
dsid = inf.split('/')[-1]
dsid = dsid.replace('.root', '')
print dsid
oldTree = f.get('nominal')
    #oldTree.SetBranchStatus("*",0)
    #for br in branch_list:
    #    oldTree.SetBranchStatus(br,1)

events = []

current = 0
for e in oldTree:
    current+=1
    if current%10000==0:
        print current

    if e.total_leptons!=2: continue
    if abs(e.total_charge)!=2: continue

    k = {}
    k['higgs_pt'] = e.higgs_pt

    k['lep_Pt_0'] = e.lep_Pt_0
    k['lep_Eta_0'] = e.lep_Eta_0
    k['lep_Phi_0'] = e.lep_Phi_0
    k['lep_E_0'] = e.lep_E_0

    k['lep_Pt_1'] = e.lep_Pt_1
    k['lep_Eta_1'] = e.lep_Eta_1
    k['lep_Phi_1'] = e.lep_Phi_1
    k['lep_E_1'] = e.lep_E_1

    k['MET'] = e.MET_RefFinal_et
    k['MET_phi'] = e.MET_RefFinal_phi
    
    if e.nJets_OR_T==4:
        n = 0
        phi0 = 0
        for i in e.selected_jets_T:
            k['jet_Pt_'+str(n)] = e.m_jet_pt[i]
            k['jet_Eta_'+str(n)] = e.m_jet_eta[i]
            k['jet_E_'+str(n)] = e.m_jet_E[i]

            if n==0:
                phi0 = e.m_jet_phi[i]
            else:
                new_phi = e.m_jet_phi[i]-phi0
                if new_phi>math.pi:
                    new_phi = new_phi - 2*math.pi
                if new_phi<-math.pi:
                    new_phi = new_phi + 2*math.pi

                k['jet_Phi_'+str(n)] = new_phi

            n+=1
    else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(sys.argv[2])
