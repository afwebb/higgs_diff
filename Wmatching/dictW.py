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
#def WTopDict3l(nom, topIdx0, topIdx1, topScore, match=-1):
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

    k['dR_lW_t0'] = lepW.DeltaR(top0) 
    k['MlWt0'] = (lepW+top0).M()                                                                                    
    k['dR_lW_t1'] = lepW.DeltaR(top1)                                                                                  
    k['MlWt1'] = (lepW+top1).M() 

    k['dR_lT0_t0'] = lepT0.DeltaR(top0) 
    k['MlT0t0'] = (lepT0+top0).M()
    k['dR_lT0_t1'] = lepT0.DeltaR(top1)  
    k['MlT0t1'] = (lepT0+top1).M()

    k['dR_lT1_t0'] = lepT1.DeltaR(top0)
    k['MlT1t0'] = (lepT1+top0).M()                                                                                     
    k['dR_lT1_t1'] = lepT1.DeltaR(top1)                                                                                  
    k['MlT1t1'] = (lepT1+top1).M()

    k['dR_lW_lT0lT1'] = lepW.DeltaR(lepT0+lepT1)
    k['dR_lT0t0_lT1t1'] = (lepT0+top0).DeltaR(lepT1+top1)
    k['dR_lT0t1_lT1t0'] = (lepT0+top1).DeltaR(lepT1+top0)

    k['MlT0lT1t0t1'] = (lepT0+lepT1+top0+top1).M()
    k['PtlT0lT1t0t1'] = (lepT0+lepT1+top0+top1).Pt()
    k['dR_lW_lT0lT1t0t1'] = (lepT0+lepT1+top0+top1).DeltaR(lepW)

    k['dPhi_lW_met'] = lepW.Phi()-met.Phi()
    k['dPhi_lT0_met'] = lepT0.Phi()-met.Phi()
    k['dPhi_lT1_met'] = lepT1.Phi()-met.Phi()

    k['met'] = met.Pt()
    k['topScore'] = topScore     
    k['HT'] = nom.HT
    k['nJets_OR'] = nom.nJets_OR                                                                                           
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70

    return k

def WTopDict2lSS(nom, lepIdx, topIdx0, topIdx1, topScore, match=-1):                                                 
#def WTopDict2lSS(nom, topIdx0, topIdx1, topScore, match=-1): 
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
    k['lep_Phi_W'] = lepW.Phi()-met.Phi()
    k['lep_Phi_T'] = lepT.Phi()-met.Phi()
    
    k['top_Pt_0'] = top0.Pt()                                                                                         
    k['top_Pt_1'] = top1.Pt()
    k['top_Eta_0'] = top0.Eta()                                                                                            
    k['top_Eta_1'] = top1.Eta()
    k['top_Phi_0'] = top0.Phi()-met.Phi()
    k['top_Phi_1'] = top1.Phi()-met.Phi()

    k['dR_lW_t0'] = lepW.DeltaR(top0)                                                                                    
    k['MlWt0'] = (lepW+top0).M()                                                                                      
    k['dR_lW_t1'] = lepW.DeltaR(top1)                                                                                    
    k['MlWt1'] = (lepW+top1).M()
    
    k['dR_lT_t0'] = lepT.DeltaR(top0)                                                                                 
    k['MlTt0'] = (lepT+top0).M()                                                                                    
    k['dR_lT_t1'] = lepT.DeltaR(top1)
    k['MlTt1'] = (lepT+top1).M()

    k['dR_lW_lTt0'] = lepW.DeltaR(top0+lepT)
    k['dR_lW_lTt1'] = lepW.DeltaR(top1+lepT)
    k['dR_lW_lTt0t1'] = lepW.DeltaR(lepT+top0+top1)
                                                                                                               
    k['met'] = met.Pt()                                                                                             
    k['topScore'] = topScore                                                                                      
    k['HT'] = nom.HT
    k['nJets_OR'] = nom.nJets_OR
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70
    
    return k

def fourVecWTopDict2lSS(nom, topIdx0, topIdx1, topScore, match=-1):

    k = {}                                                                                           
    k['lep_Pt_0'] = nom.lep_Pt_0                                                                               
    k['lep_Eta_0'] = nom.lep_Eta_0                                                                          
    k['lep_Phi_0'] = nom.lep_Phi_0                                                                            
    k['lep_Pt_1'] = nom.lep_Pt_1                                                                             
    k['lep_Eta_1'] = nom.lep_Eta_1                                                                             
    k['lep_Phi_1'] = nom.lep_Phi_1                                                                               
    k['met_met'] = nom.met_met
    k['met_phi'] = nom.met_phi
    k['nJets_OR'] = nom.nJets_OR                                                                              
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70
    k['bjet_pt_0'] = nom.jet_pt[topIdx0]
    k['bjet_eta_0'] = nom.jet_eta[topIdx0]
    k['bjet_phi_0'] = nom.jet_phi[topIdx0]
    k['bjet_DL1r_0'] = nom.jet_DL1r[topIdx0]                                                             
    k['bjet_pt_1'] = nom.jet_pt[topIdx1]
    k['bjet_eta_1'] = nom.jet_eta[topIdx1]
    k['bjet_phi_1'] = nom.jet_phi[topIdx1]
    k['bjet_DL1r_1'] = nom.jet_DL1r[topIdx1]
    k['topScore'] = topScore

    if match!=-1:
        if abs(nom.lep_Parent_0)==24:
            k['match']=1
        elif abs(nom.lep_Parent_1)==24:
            k['match']=0
        else:
            print('no W match found')                                                                              
            return

    return k

def fourVecWTopDict3l(nom, topIdx0, topIdx1, topScore, match=-1):

    k = {}
    k['lep_Pt_0'] = nom.lep_Pt_0
    k['lep_Eta_0'] = nom.lep_Eta_0
    k['lep_Phi_0'] = nom.lep_Phi_0
    k['lep_Pt_1'] = nom.lep_Pt_1
    k['lep_Eta_1'] = nom.lep_Eta_1
    k['lep_Phi_1'] = nom.lep_Phi_1
    k['lep_Pt_2'] = nom.lep_Pt_2                                                                                    
    k['lep_Eta_2'] = nom.lep_Eta_2                                                                                 
    k['lep_Phi_2'] = nom.lep_Phi_2 
    k['met_met'] = nom.met_met
    k['met_phi'] = nom.met_phi
    k['nJets_OR'] = nom.nJets_OR
    k['nJets_OR_DL1r_70'] = nom.nJets_OR_DL1r_70
    k['bjet_pt_0'] = nom.jet_pt[topIdx0]
    k['bjet_eta_0'] = nom.jet_eta[topIdx0]
    k['bjet_phi_0'] = nom.jet_phi[topIdx0]
    k['bjet_DL1r_0'] = nom.jet_DL1r[topIdx0]
    k['bjet_pt_1'] = nom.jet_pt[topIdx1]
    k['bjet_eta_1'] = nom.jet_eta[topIdx1]
    k['bjet_phi_1'] = nom.jet_phi[topIdx1]
    k['bjet_DL1r_1'] = nom.jet_DL1r[topIdx1]
    k['topScore'] = topScore

    if match!=-1:
        if abs(nom.lep_Parent_1)==24:
            k['match']=1
        elif abs(nom.lep_Parent_2)==24:
            k['match']=0
        else:
            print('no W match found')
            return

    return k
