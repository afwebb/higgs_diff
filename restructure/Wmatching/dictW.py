# Defines dictionaries to be used for higgs reconstruction algorithms, which try to identify the decay products of the Higgs

import ROOT
from rootpy.vector import LorentzVector
#from functionsMatch import lorentzVecsHiggs, lorentzVecsTop

def lorentzVecsLeps(nom, is3l):

    '''
    Higgs decays to two jets and one lepton, or two leptons. This returns lorentzVectors for each decay product candidate
    For H -> 2j, 1l case (not isF): returns jet0, jet1, met, lep0, lep1, (lep2 if is3l)
    For H -> 2l case (isF): Return met, lep0, lep1, lep2 
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


    if is3l:
        return (lep0, lep1, lep2, met)
    else:
        return (lep0, lep1, met)

def lorentzVecsTop(nom, topIdx0, topIdx1):
    '''
    Takes the indices of two jets identified to be bjets from top decay, return their LorentzVectors 
    '''

    top0 = LorentzVector()                                                                                                   
    top0.SetPtEtaPhiE(nom.jet_pt[topIdx0], nom.jet_eta[topIdx0], nom.jet_phi[topIdx0], nom.jet_e[topIdx0])
    top1 = LorentzVector()                                                                                         
    top1.SetPtEtaPhiE(nom.jet_pt[topIdx1], nom.jet_eta[topIdx1], nom.jet_phi[topIdx1], nom.jet_e[topIdx1])

    return (top0, top1)

def WTopDict3l(nom, lepIdx, topIdx0, topIdx1, topScore, match=-1):
    '''                                                                                                                      
    '''

    lep0, lep1, lep2, met = lorentzVecsLeps(nom, 1)                                                          
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)                                                             

    #Set which lepton is from the W, which from top
    lepT0 = lep0
    if lepIdx == 1:                                                                                                    
        lepW = lep1                                                                                               
        lepT1 = lep2                                                                                                 
    elif lepIdx == 2:
        lepW = lep2
        lepT1 = lep1
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0
        
    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_W'] = lepW.Pt()                  
    k['lep_Pt_T0'] = lepT0.Pt()
    k['lep_Pt_T1'] = lepT1.Pt()

    k['lep_Eta_W'] = lepW.Eta()    
    k['lep_Eta_T0'] = lepT0.Eta()                                                                                        
    k['lep_Eta_T1'] = lepT1.Eta()

    k['top_Pt_0'] = top0.Pt()   
    k['top_Pt_1'] = top1.Pt()  
    
    k['MlT0lW'] = (lepW+lepT0).M()
    k['MlWlT1'] = (lepW+lepT1).M()                                                                                       
    k['MlT0lT1'] = (lepT0+lepT1).M()

    k['dR_lT0_lW'] = lepW.DeltaR(lepT0)
    k['dR_lW_lT1'] = lepW.DeltaR(lepT1)
    k['dR_lT0_lT1'] = lepT0.DeltaR(lepT1)

    k['dRlWt0'] = lepW.DeltaR(top0) 
    k['MlWt0'] = (lepW+top0).M()                                                                                    
    k['dRlWt1'] = lepW.DeltaR(top1)                                                                                  
    k['MlWt1'] = (lepW+top1).M() 

    k['dRlT0t0'] = lepT0.DeltaR(top0) 
    k['MlT0t0'] = (lepT0+top0).M()
    k['dRlT0t1'] = lepT0.DeltaR(top1)  
    k['MlT0t1'] = (lepT0+top1).M()

    k['dRlT1t0'] = lepT1.DeltaR(top0)
    k['MlT1t0'] = (lepT1+top0).M()                                                                                     
    k['dRlT1Ht1'] = lepT1.DeltaR(top1)                                                                                  
    k['MlT1t1'] = (lepT1+top1).M()

    k['dPhi_lW_met'] = lepW.Phi()-met.Phi()
    k['dPhi_lT0_met'] = lepT0.Phi()-met.Phi()
    k['dPhi_lT1_met'] = lepT1.Phi()-met.Phi()

    k['met'] = met.Pt()
    k['topScore'] = topScore     
    k['HT'] = nom.HT

    return k

def WTopDict2lSS(nom, lepIdx, topIdx0, topIdx1, topScore, match=-1):                                                         
    '''
    '''

    lep0, lep1, met = lorentzVecsLeps(nom, 0)    
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)
    
    #Set which lepton is from the W, which from top                                                                    
    if lepIdx == 1:                                                                                                     
        lepW = lep1                                                                                                  
        lepT = lep0                                                                                                 
    elif lepIdx == 0:  
        lepW = lep0                                                                                               
        lepT = lep1                                                                                                
    else:                                                                                                           
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0
    
    k = {}                                                                                                          
    if match!=-1:                                                                                                 
        k['match'] = match
    
    k['lep_Pt_W'] = lepW.Pt()                                                                                           
    k['lep_Pt_T'] = lepT.Pt()
    k['lep_Eta_W'] = lepW.Eta()
    k['lep_Eta_T'] = lepT.Eta()                                                                                      
    
    k['top_Pt_0'] = top0.Pt()                                                                                         
    k['top_Pt_1'] = top1.Pt()
    
    k['dRlWt0'] = lepW.DeltaR(top0)                                                                                    
    k['MlWt0'] = (lepW+top0).M()                                                                                      
    k['dRlWt1'] = lepW.DeltaR(top1)                                                                                    
    k['MlWt1'] = (lepW+top1).M()
    
    k['dRlTt0'] = lepT.DeltaR(top0)                                                                                 
    k['MlTt0'] = (lepT+top0).M()                                                                                    
    k['dRlTt1'] = lepT.DeltaR(top1)
    k['MlTt1'] = (lepT+top1).M()
    
    k['dPhi_lW_met'] = lepW.Phi()-met.Phi()
    k['dPhi_lT_met'] = lepT.Phi()-met.Phi()                                                                         
                                                                                                               
    k['met'] = met.Pt()                                                                                             
    k['topScore'] = topScore                                                                                      
    k['HT'] = nom.HT
    
    return k
