# Defines dictionaries to be used for higgs reconstruction algorithms, which try to identify the decay products of the Higgs

import ROOT
from rootpy.vector import LorentzVector
from dictHiggs import lorentzVecsTop, lorentzVecsHiggs

def decayDict(nom, score3lF, score3lS, topIdx0, topIdx1, topScore, match=-1):
    '''      
    Returns a dict for predicting whether the Higgs decayed into two leptons (F - fully leptonic) or one (S - semi-leptonic)
    Takes a TTree, the indices of top jets, 
    '''

    met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, 0, 0, 1, 1)                                                          
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)                                                             

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_0'] = lep0.Pt()
    k['lep_Pt_1'] = lep1.Pt()
    k['lep_Pt_2'] = lep2.Pt()

    k['lep_Eta_0'] = lep0.Eta()                                                                                         
    k['lep_Eta_1'] = lep1.Eta()
    k['lep_Eta_2'] = lep2.Eta()

    k['lep_Phi_0'] = lep0.Phi() - met.Phi()   
    k['lep_Phi_1'] = lep1.Phi() - met.Phi()                                                                          
    k['lep_Phi_2'] = lep2.Phi() - met.Phi()

    k['Ml0l1'] = (lep0+lep1).M()
    k['Ml0l2'] = (lep0+lep2).M()
    k['Ml1l2'] = (lep1+lep2).M()

    k['dRl0l1'] = lep0.DeltaR(lep1)
    k['dRl0l2'] = lep0.DeltaR(lep2)                                                                                 
    k['dRl1l2'] = lep1.DeltaR(lep2)

    k['met'] = nom.met_met
    k['Ml0l1met'] = (lep0+lep1+met).M()
    k['Ml0l2met'] = (lep0+lep2+met).M()
    k['Ml1l2met'] = (lep1+lep2+met).M()
   
    k['dRl0t0'] = lep0.DeltaR(top0) 
    k['Ml0t0'] = (lep0+top0).M()
    k['dRl0t1'] = lep0.DeltaR(top1)  
    k['Ml0t1'] = (lep0+top1).M()

    k['dRl0t0'] = lep1.DeltaR(top0)                                                                                     
    k['Ml0t0'] = (lep1+top0).M()                                                                                     
    k['dRl0t1'] = lep1.DeltaR(top1)                                                                                   
    k['Ml0t1'] = (lep1+top1).M()

    k['dRl1t0'] = lep2.DeltaR(top0)                                                                                        
    k['Ml1t0'] = (lep2+top0).M()                                                                                      
    k['dRl1t1'] = lep2.DeltaR(top1)                                                                                 
    k['Ml1t1'] = (lep2+top1).M()
 
    k['HT_jets'] = nom.HT_jets
    k['nJets_OR'] = nom.nJets_OR
    k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85
    k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60
    k['total_charge'] = nom.total_charge
    k['trilep_type'] = nom.trilep_type
    
    k['topScore'] = topScore
    k['score3lF'] = score3lF
    k['score3lS'] = score3lS

    return k
