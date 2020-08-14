# Defines dictionaries to be used for top match algorithm

import ROOT
from rootpy.vector import LorentzVector

def lorentzVecs(nom, jetIdx0, jetIdx1, is3l):

    #initialize met, lepton, and jet four vectors                                                                     
    # Return jet0, jet1, met, lep0, lep1, (lep2 if is3l)

    met = LorentzVector()
    met.SetPtEtaPhiE(nom.met_met, 0, nom.met_phi, nom.met_met)

    lep0 = LorentzVector()
    lep0.SetPtEtaPhiE(nom.lep_Pt_0, nom.lep_Eta_0, nom.lep_Phi_0, nom.lep_E_0)

    lep1 = LorentzVector()
    lep1.SetPtEtaPhiE(nom.lep_Pt_1, nom.lep_Eta_1, nom.lep_Phi_1, nom.lep_E_1)

    if is3l:
        lep2 = LorentzVector()
        lep2.SetPtEtaPhiE(nom.lep_Pt_2, nom.lep_Eta_2, nom.lep_Phi_2, nom.lep_E_2)

    jet0 = LorentzVector()                                                                                                
    jet0.SetPtEtaPhiE(nom.jet_pt[jetIdx0], nom.jet_eta[jetIdx0], nom.jet_phi[jetIdx0], nom.jet_e[jetIdx0])

    jet1 = LorentzVector()                                                                                                   
    jet1.SetPtEtaPhiE(nom.jet_pt[jetIdx1], nom.jet_eta[jetIdx1], nom.jet_phi[jetIdx1], nom.jet_e[jetIdx1])

    if is3l:
        return (jet0, jet1, met, lep0, lep1, lep2)
    else:
        return (jet0, jet1, met, lep0, lep1)

def topDictFlat2lSS(nom, jetIdx0, jetIdx1, match=-1):#, lep0, lep1, met, jet0_DL1r, jet1_DL1r, match=-1):
    k = {}

    if match!=-1:
        k['match'] = match

    #initialize met, lepton four vectors                                                                
    jet0, jet1, met, lep0, lep1 = lorentzVecs(nom, jetIdx0, jetIdx1, 0)

    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['dRjj'] = jet0.DeltaR(jet1)
    #k['Ptjj'] = (jet0+jet1).Pt()
    k['Mjj'] = (jet0+jet1).M()

    k['dRlj00'] = lep0.DeltaR(jet0)
    k['Mlj00'] = (lep0+jet0).M()

    k['dRlj01'] = lep0.DeltaR(jet1)
    k['Mlj01'] = (lep0+jet1).M()

    k['dRlj10'] = lep1.DeltaR(jet0)
    k['Mlj10'] = (lep1+jet0).M()

    k['dRlj11'] = lep1.DeltaR(jet1)
    k['Mlj11'] = (lep1+jet1).M()

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    k['dRjjl0'] = (jet0+jet1).DeltaR(lep0)
    k['dRjjl1'] = (jet0+jet1).DeltaR(lep1)

    #k['Mjjl0'] = (jet0+jet1+lep0).M()
    #k['Mjjl1'] = (jet0+jet1+lep1).M()

    #k['dRj0l0met'] = jet0.DeltaR(lep0+met)
    #k['dRj0l1met'] = jet0.DeltaR(lep1+met)

    #k['dRj1l0met'] = jet1.DeltaR(lep0+met)
    #k['dRj1l1met'] = jet1.DeltaR(lep1+met)

    k['dRj0met'] = jet0.DeltaR(met)
    k['dRj1met'] = jet1.DeltaR(met)

    k['HT_lep'] = nom.HT_lep
    k['HT_jets'] = nom.HT_jets

    #k['dRjjmet'] = (jet0+jet1).DeltaR(met)
    return k
    
def topDictFourVec2lSS(nom, jetIdx0, jetIdx1, match=-1):#(jet0, jet1, lep0, lep1, met, jet0_DL1r, jet1_DL1r, match=-1):
    p = {}
    if match!=-1:
        p['match'] = match

    p['jet_Pt_0'] = nom.jet_pt[jetIdx0]
    p['jet_Eta_0'] = nom.jet_eta[jetIdx0]
    p['jet_Phi_0'] = nom.jet_phi[jetIdx0]
    p['jet_E_0'] = nom.jet_e[jetIdx0]
    
    p['jet_Pt_1'] = nom.jet_pt[jetIdx1]                                                                                      
    p['jet_Eta_1'] = nom.jet_eta[jetIdx1]                                                                             
    p['jet_Phi_1'] = nom.jet_phi[jetIdx1]                                                                                   
    p['jet_E_1'] = nom.jet_e[jetIdx1]

    p['lep_Pt_0'] = nom.lep_Pt_0
    p['lep_Eta_0'] = nom.lep_Eta_0
    p['lep_Phi_0'] = nom.lep_Phi_0
    p['lep_E_0'] = nom.lep_E_0

    p['lep_Pt_1'] = nom.lep_Pt_1
    p['lep_Eta_1'] = nom.lep_Eta_1
    p['lep_Phi_1'] = nom.lep_Phi_1
    p['lep_E_1'] = nom.lep_E_1

    p['met'] = nom.met_met
    p['met_phi'] = nom.met_phi
    
    p['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]
    p['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    return p

def topDictFlat3l(nom, jetIdx0, jetIdx1, match=-1):#(jet0, jet1, lep0, lep1, lep2, met, jet0_DL1r, jet1_DL1r, match=-1):
    k = {}

    if match!=-1:
        k['match'] = match

    jet0, jet1, met, lep0, lep1, lep2 = lorentzVecs(nom, jetIdx0, jetIdx1, 1)

    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['dRjj'] = jet0.DeltaR(jet1)
    #k['Ptjj'] = (jet0+jet1).Pt()
    k['Mjj'] = (jet0+jet1).M()

    k['dRlj00'] = lep0.DeltaR(jet0)
    k['Mlj00'] = (lep0+jet0).M()

    k['dRlj01'] = lep0.DeltaR(jet1)
    k['Mlj01'] = (lep0+jet1).M()

    k['dRlj10'] = lep1.DeltaR(jet0)
    k['Mlj10'] = (lep1+jet0).M()

    k['dRlj11'] = lep1.DeltaR(jet1)
    k['Mlj11'] = (lep1+jet1).M()

    k['dRlj20'] = lep2.DeltaR(jet0)
    k['Mlj20'] = (lep2+jet0).M()

    k['dRlj21'] = lep2.DeltaR(jet1)
    k['Mlj21'] = (lep2+jet1).M()

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]

    k['dRj0met'] = jet0.DeltaR(met)
    k['dRj1met'] = jet1.DeltaR(met)

    k['HT_lep'] = nom.HT_lep   
    k['HT_jets'] = nom.HT_jets

    #k['met'] = met.Pt()
    #k['met_phi']= met.Phi()

    return k

def topDictFourVec3l(nom, jetIdx0, jetIdx1, match=-1):#(jet0, jet1, lep0, lep1, lep2, met, jet0_DL1r, jet1_DL1r, match=-1):
    p = {}
    if match!=-1:
        p['match'] = match

    p['jet_Pt_0'] = nom.jet_pt[jetIdx0]                                                                                      
    p['jet_Eta_0'] = nom.jet_eta[jetIdx0]
    p['jet_Phi_0'] = nom.jet_phi[jetIdx0]                                                                                  
    p['jet_E_0'] = nom.jet_e[jetIdx0]                                                                                        

    p['jet_Pt_1'] = nom.jet_pt[jetIdx1]                                                                                     
    p['jet_Eta_1'] = nom.jet_eta[jetIdx1]                                                                                
    p['jet_Phi_1'] = nom.jet_phi[jetIdx1]
    p['jet_E_1'] = nom.jet_e[jetIdx1]
    
    p['lep_Pt_0'] = nom.lep_Pt_0
    p['lep_Eta_0'] = nom.lep_Eta_0                                                                                       
    p['lep_Phi_0'] = nom.lep_Phi_0                                                                                      
    p['lep_E_0'] = nom.lep_E_0
    
    p['lep_Pt_1'] = nom.lep_Pt_1                                                                                           
    p['lep_Eta_1'] = nom.lep_Eta_1
    p['lep_Phi_1'] = nom.lep_Phi_1
    p['lep_E_1'] = nom.lep_E_1

    p['lep_Pt_2'] = nom.lep_Pt_2
    p['lep_Eta_2'] = nom.lep_Eta_2                                                                                        
    p['lep_Phi_2'] = nom.lep_Phi_2                                                                                      
    p['lep_E_2'] = nom.lep_E_2

    p['met'] = nom.met_met                                                                                           
    p['met_phi'] = nom.met_phi
    
    p['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    p['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]
    
    return p
