import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math
import higgsCandidate

nj = 6
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

from collections import namedtuple
parttype = namedtuple('parttype', ['barcode', 'pdgid', 'status', 'eta', 'phi', 'pt', 'parents', 'children'])

def make_partdict(e):
    rv = dict(zip(e.m_truth_barcode,
                 (parttype(*_2) for
                  _2 in zip(e.m_truth_barcode,
                            e.m_truth_pdgId,
                            e.m_truth_status,
                            e.m_truth_eta,
                            e.m_truth_phi,
                            e.m_truth_pt,
                            e.m_truth_parents,
                            e.m_truth_children)
                 )
                 ))
    return rv

def drCheck(eta, phi, truth_eta, truth_phi, cut):
    dr = math.sqrt( (phi-truth_phi)**2 + (eta-truth_eta)**2 )
    #print('DR', dr, pt/part.pt-1)
    #print('DR', dr)
    return dr < cut

current = 0
nMatch = 0

for e in oldTree:
    current+=1
    if current%10000==0:
        print(current)
    #if current%100000==0:
    #    break
 
    #if e.is2LSS0Tau==0 and e.is2LSS1Tau==0: continue
    if e.dilep_type==0: continue
    if abs(e.total_charge)!=2: continue
    #if e.nJets_OR_T<nj: continue
    if e.higgsDecayMode != 3: continue

    k = {}
    k['higgs_pt'] = e.higgs_pt

    truth_dict = make_partdict(e)

    lepPts = [e.lep_Pt_0, e.lep_Pt_1]
    lepEtas = [e.lep_Eta_0, e.lep_Eta_1]
    lepPhis = [e.lep_Phi_0, e.lep_Phi_1]
    lepEs = [e.lep_E_0, e.lep_E_1]
    lepIDs = [e.lep_ID_0, e.lep_ID_1]

    #lepton matching
    lepMatch = -1
    for i in range(2):
        for a in truth_dict:
            if abs(truth_dict[a].pdgid) == abs(lepIDs[i]): 
                if drCheck(lepEtas[i], lepPhis[i], truth_dict[a].eta, truth_dict[a].phi, 0.01):
                    p = truth_dict[a].parents[0]
                    terminal = False
                    while not terminal:
                        if p in truth_dict:
                            a = p
                            try:
                                p = truth_dict[a].parents[0]
                            except:
                                terminal = True
                        else: terminal = True
                    if truth_dict[a].pdgid == 25: lepMatch = i

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
        
    else:
    #    print("no lepton match found")
        continue

    k['MET'] = e.MET_RefFinal_et
    k['MET_phi'] = calc_phi(phi_0, e.MET_RefFinal_phi)

    #jet truth matching
    higgsID = 0
    for x in truth_dict:
        if truth_dict[x].pdgid==25: higgsID = x

    jetCands = []
    jetTest = []
    Ws = truth_dict[higgsID].children
    for w in Ws:
        for child in truth_dict[w].children:
            if child in truth_dict: 
                ch = truth_dict[child]
            else: 
                continue
            if abs(ch.pdgid) in range(1,5): 
                jetTest.append(child)
            jetCands.append(child)#=[*jetCands, *truth_dict[w].children]
        
    if len(jetTest)!=2:
        continue

    jetMatches = []
    for j in jetCands:
        if j not in truth_dict:
            print('not found')
            continue
        if abs(truth_dict[j].pdgid) not in range(1, 5):
            continue
        else:
            for idx in range(len(e.m_jet_pt)):#la[b'selected_jets'][i]:
                jet_pt = e.m_jet_pt[idx]
                jet_eta = e.m_jet_eta[idx]
                jet_phi = e.m_jet_phi[idx]
                jet_flav = e.m_jet_flavor_truth_label_ghost[idx]

                dr = math.sqrt((jet_phi-truth_dict[j].phi)**2+(jet_eta-truth_dict[j].eta)**2)
                if dr<0.3 and abs(jet_flav)==abs(truth_dict[j].pdgid):
                    jetMatches.append(idx)

    if len(jetMatches)!=2: 
        continue 
    else: 
        nMatch+=1

    n = 0
    for i in jetMatches:#e.nJets_OR_T):

        k['jet_Pt_'+str(n)] = e.m_jet_pt[i]
        k['jet_Eta_'+str(n)] = e.m_jet_eta[i]
        k['jet_E_'+str(n)] = e.m_jet_E[i]
        k['jet_Phi_'+str(n)] = calc_phi(phi_0, e.m_jet_phi[i])
        
        n+=1

    #k['rough_pt'], k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e)
    #_, k['lepJetCat'] = higgsCandidate.calcHiggsCandidate(e) 
    #else: continue

    events.append(k)

    
dFrame = pd.DataFrame(events)
dFrame.to_csv(sys.argv[2])
