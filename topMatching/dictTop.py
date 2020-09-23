'''
Defines input variables to be used for top matching algorithm. Includes "fourVec" and "flat" variations, which include low level variable (pt, eta, phi) for pairs of jets, and higher level variable (dR, M, Pt) for both 2lSS and 3l Channels
'''

import ROOT
from rootpy.vector import LorentzVector

def lorentzVecs(nom, jetIdx0, jetIdx1, is3l):
    '''
    Initialize met, lepton, and jet lorentz vectors       
    Return jet0, jet1, met, lep0, lep1, (lep2 if is3l)
    '''

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

def topDict2lSS(nom, jetIdx0, jetIdx1, match=-1):
    '''
    Create a dictionary with lepton, jet kinematics to distinguish jets from tops from background jets for 2lSS channel
    Truth label "match" =1 if both jets decayed from tops, =0 otherwise
    '''

    k = {}

    if match!=-1:
        k['match'] = match

    #initialize met, lepton four vectors                                                                
    jet0, jet1, met, lep0, lep1 = lorentzVecs(nom, jetIdx0, jetIdx1, 0)

    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['lep_Pt_0'] = lep0.Pt()
    k['lep_Pt_1'] = lep1.Pt()

    k['jet_Eta_0'] = jet0.Eta()
    k['jet_Eta_1'] = jet1.Eta()

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    #k['Ptj0j1'] = (jet0+jet1).Pt()
    k['Mj0j1'] = (jet0+jet1).M()

    k['dR_l0_j0'] = lep0.DeltaR(jet0)
    k['dR_l0_j1'] = lep0.DeltaR(jet1)
    k['dR_l1_j0'] = lep1.DeltaR(jet0)
    k['dR_l1_j1'] = lep1.DeltaR(jet1)

    k['Ml0j0'] = (lep0+jet0).M()
    k['Ml0j1'] = (lep0+jet1).M()
    k['Ml1j0'] = (lep1+jet0).M()
    k['Ml1j1'] = (lep1+jet1).M()

    #k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]
    #k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]
    rDL1r = [len(nom.jet_DL1r)-sorted(nom.jet_DL1r).index(x) for x in nom.jet_DL1r]
    k['jet_rankDL1r_0'] = rDL1r[jetIdx0]
    k['jet_rankDL1r_1'] = rDL1r[jetIdx1]
    #k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85                                                                  
    #k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    #k['dR_j0j1_l0'] = (jet0+jet1).DeltaR(lep0)
    #k['dR_j0j1_l1'] = (jet0+jet1).DeltaR(lep1)

    k['dRj0l0j1l1'] = (jet0+lep0).DeltaR(jet1+lep1)
    k['dRj0l1j1l0'] = (jet0+lep1).DeltaR(jet1+lep0)

    k['Ptj0j1l0l1met'] = (jet0+jet1+lep0+lep1+met).Pt()
    k['Mj0j1l0l1met'] = (jet0+jet1+lep0+lep1+met).M()
    #k['Mj0j1l0'] = (jet0+jet1+lep0).M()
    #k['Mj0j1l1'] = (jet0+jet1+lep1).M()

    k['dPhi_j0_met'] = jet0.Phi() - met.Phi()
    k['dPhi_j1_met'] = jet1.Phi() - met.Phi()

    #k['HT_lep'] = nom.HT_lep
    k['HT_jets'] = nom.HT_jets
    k['nJets_OR'] = nom.nJets_OR
    k['met'] = nom.met_met

    return k
    
def topDictFourVec2lSS(nom, jetIdx0, jetIdx1, match=-1):#(jet0, jet1, lep0, lep1, met, jet0_DL1r, jet1_DL1r, match=-1):
    '''
    Return a dictionary of low level varibles for a jet pairing. match = 1 -> both jets are b-jets from top decay, match=0 
    otherwise
    '''

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
    p['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85
    p['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    p['HT_lep'] = nom.HT_lep                                                      
    p['HT_jets'] = nom.HT_jets                                                                            
    p['nJets_OR'] = nom.nJets_OR

    return p

def topDict3l(nom, jetIdx0, jetIdx1, match=-1):
    '''                                                                                                                     
    Create a dictionary with lepton, jet kinematics to distinguish jets from tops from background jets              
    Truth label "match" =1 if both jets decayed from tops, =0 otherwise                                                
    '''  
    
    k = {}

    if match!=-1:
        k['match'] = match

    jet0, jet1, met, lep0, lep1, lep2 = lorentzVecs(nom, jetIdx0, jetIdx1, 1)

    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['jet_Eta_0'] = jet0.Eta()
    k['jet_Eta_1'] = jet1.Eta()

    k['lep_Pt_0'] = lep0.Pt()                                                                                       
    k['lep_Pt_1'] = lep1.Pt() 
    k['lep_Pt_2'] = lep2.Pt() 

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    #k['Ptj0j1'] = (jet0+jet1).Pt()
    k['Mj0j1'] = (jet0+jet1).M()

    k['dR_l0_j0'] = lep0.DeltaR(jet0)
    k['dR_l1_j0'] = lep1.DeltaR(jet0)
    k['dR_l2_j0'] = lep2.DeltaR(jet0)

    k['dR_l0_j1'] = lep0.DeltaR(jet1)
    k['dR_l1_j1'] = lep1.DeltaR(jet1)
    k['dR_l2_j1'] = lep2.DeltaR(jet1)

    k['Ml0j0'] = (lep0+jet0).M()
    k['Ml1j0'] = (lep1+jet0).M()
    k['Ml2j0'] = (lep2+jet0).M()

    k['Ml0j1'] = (lep0+jet1).M()                                                                                       
    k['Ml1j1'] = (lep1+jet1).M()                                                                                      
    k['Ml2j1'] = (lep2+jet1).M()

    #Both tops -> l+j. dR(tt) should be large. Consider dR of all l+j combos? Maybe Pt(lj,lj)?
    k['dR_j0l0_j1l1'] = (jet0+lep0).DeltaR(jet1+lep1)
    k['dR_j0l0_j1l2'] = (jet0+lep0).DeltaR(jet1+lep2) 
    k['dR_j0l1_j1l0'] = (jet0+lep1).DeltaR(jet1+lep0)                                               
    k['dR_j0l2_j1l0'] = (jet0+lep2).DeltaR(jet1+lep0)

    #Could be h->2l, t->lb, t->jjb. Lep 1 is closer to lep 2
    #k['dR_j0_l0l1'] = jet0.DeltaR(lep0+lep1)
    #k['dR_j1_l0l1'] = jet1.DeltaR(lep0+lep1)
    #k['dR_j0_l0l2'] = jet0.DeltaR(lep0+lep2)                                                                               
    #k['dR_j1_l0l2'] = jet1.DeltaR(lep0+lep2)

    #k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                         
    #k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]
    rDL1r = [len(nom.jet_DL1r)-sorted(nom.jet_DL1r).index(x) for x in nom.jet_DL1r]                                    
    k['jet_rankDL1r_0'] = rDL1r[jetIdx0]                                                                                  
    k['jet_rankDL1r_1'] = rDL1r[jetIdx1]

    k['Ptj0j1l0l1l2met'] = (jet0+jet1+lep0+lep1+lep2+met).Pt()
    k['Mtj0j1l0l1l2met'] = (jet0+jet1+lep0+lep1+lep2+met).M()

    k['dPhi_j0_met'] = jet0.Phi() - met.Phi()                                                                         
    k['dPhi_j1_met'] = jet1.Phi() - met.Phi()                                                                     
    
    #k['HT_lep'] = nom.HT_lep   
    k['HT_jets'] = nom.HT_jets
    k['nJets_OR'] = nom.nJets_OR
    k['met'] = nom.met_met

    #k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85 
    #k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

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

    p['nJets_OR'] = nom.nJets_OR
    p['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85                                                                             
    p['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    p['HT_lep'] = nom.HT_lep                 
    p['HT_jets'] = nom.HT_jets
    
    return p
