# Defines dictionaries to be used for higgs reconstruction algorithms, which try to identify the decay products of the Higgs

import ROOT
from rootpy.vector import LorentzVector
#from functionsMatch import lorentzVecsHiggs, lorentzVecsTop

def lorentzVecsHiggs(nom, jetIdx0, jetIdx1, is3l, isF):

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

    if isF:
        return (met, lep0, lep1, lep2)
    else:
        jet0 = LorentzVector()
        jet0.SetPtEtaPhiE(nom.jet_pt[jetIdx0], nom.jet_eta[jetIdx0], nom.jet_phi[jetIdx0], nom.jet_e[jetIdx0])

        jet1 = LorentzVector()
        jet1.SetPtEtaPhiE(nom.jet_pt[jetIdx1], nom.jet_eta[jetIdx1], nom.jet_phi[jetIdx1], nom.jet_e[jetIdx1])

        if is3l:
            return (jet0, jet1, met, lep0, lep1, lep2)
        else:
            return (jet0, jet1, met, lep0, lep1)

def lorentzVecsTop(nom, topIdx0, topIdx1):
    '''
    Takes the indices of two jets identified to be bjets from top decay, return their LorentzVectors 
    '''

    top0 = LorentzVector()                                                                                                   
    top0.SetPtEtaPhiE(nom.jet_pt[topIdx0], nom.jet_eta[topIdx0], nom.jet_phi[topIdx0], nom.jet_e[topIdx0])
    top1 = LorentzVector()                                                                                         
    top1.SetPtEtaPhiE(nom.jet_pt[topIdx1], nom.jet_eta[topIdx1], nom.jet_phi[topIdx1], nom.jet_e[topIdx1])

    return (top0, top1)

def higgsDict2lSS(nom, jetIdx0, jetIdx1, lepIdx, match=-1):#, lep, met, jet1_MV2c10, jet2_MV2c10, lepO, jet1_numTrk, jet2_numTrk, match=-1):
    '''
    Takes a ttree, two jet indices, and which of the two leptons is considered part of the Higgs candidate. 
    Returns a dict for identifying Higgs decay products
    Give match if using for training. =1 if these jets, leptons are truth matched to the Higgs, 0 if not
    '''

    #Initialize four-vectors of decay product candidates
    jet0, jet1, met, lep0, lep1 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 0, 0)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs
    if lepIdx == 0:
        lepH = lep0
        lepT = lep1
    elif lepIdx == 1:
        lepH = lep1
        lepT = lep0
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 0 or 1")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['lep_Pt_T'] = lepT.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['lep_Eta_H'] = lepH.Eta()
    k['lep_Eta_T'] = lepT.Eta()
    k['jet_Eta_0'] = jet0.Eta()
    k['jet_Eta_1'] = jet1.Eta()

    k['lep_Phi_H'] = lepH.Phi() - met.Phi()
    k['lep_Phi_T'] = lepT.Phi() - met.Phi()                                                                          
    k['jet_Phi_0'] = jet0.Phi() - met.Phi()
    k['jet_Phi_1'] = jet1.Phi() - met.Phi()

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    #k['Ptj0j1'] = (jet0+jet1).Pt()
    k['Mj0j1'] = (jet0+jet1).M()

    k['dR_lH_j0'] = lepH.DeltaR(jet0)
    #k['PtlHj0'] = (lepH+jet0).Pt()
    k['MlHj0'] = (lepH+jet0).M()

    k['dR_lH_j1'] = lepH.DeltaR(jet1)
    #k['PtlHj1'] = (lepH+jet1).Pt()
    k['MlHj1'] = (lepH+jet1).M()

    k['dR_lT_j0'] = lepH.DeltaR(jet0)
    #k['PtlHj0'] = (lepH+jet0).Pt()                                                                                       
    k['MlTj0'] = (lepH+jet0).M()
    
    k['dR_lT_j1'] = lepT.DeltaR(jet1)
    #k['PtlHj1'] = (lepH+jet1).Pt()                                                                                   
    k['MlTj1'] = (lepT+jet1).M()

    k['dR_j0j1_l'] = (jet0 + jet1).DeltaR(lepH)
    k['Mj0j1lH'] = (jet0+jet1+lepH).M()

    #k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                
    #k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]
    #k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85                                                                       
    #k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    k['dR_jj_lT'] = (jet0+jet1).DeltaR(lepT)
    #k['Mj0lT'] = (jet0+lepT).M()
    #k['Mj1lT'] = (jet1+lepT).M()
    k['Mj0j1lT'] = (jet0+jet1+lepT).M()

    k['dR_j0j1lH_lO'] = (jet0+jet1+lepH).DeltaR(lepT)
    k['dPhi_j0j1lH_met'] = (jet0+jet1+lepH).Phi() - met.Phi()

    k['MET'] = met.Pt()
    k['HT'] = nom.HT

    return k

def higgsDict3lF(nom, lepIdx, match=-1):
    '''                                                                                                                      
    Takes a ttree, index of which of the leptons is considered part of the Higgs candidate. Must be 1 or 2 (lep0 is from top)
    Returns a dict for identifying Higgs decay products                     
    Gives match if using for training. =1 if these jets, leptons are truth matched to the Higgs, 0 if not                  
    ''' 

    k = {}
    met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, 0, 0, 1, 1)

    #Assign leptons to parents
    lepH0 = lep0
    if lepIdx == 1:
        lepH1, lepT = lep1, lep2
    elif lepIdx == 2:
        lepH1, lepT, = lep2, lep1
    else:                                                                                                             
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0

    if match!=-1:
        k['match'] = match

    k['lep_Pt_H1'] = lepH1.Pt()
    k['lep_Pt_H0'] = lepH0.Pt()
    k['lep_Pt_T'] = lepT.Pt()

    k['lep_Eta_H1'] = lepH1.Eta()
    k['lep_Eta_H0'] = lepH0.Eta()
    k['lep_Eta_T'] = lepT.Eta()

    k['MlH0lH1'] = (lepH1+lepH0).M()
    k['MlH1lT'] = (lepH1+lepT).M()
    k['MlH0lT'] = (lepH0+lepT).M()
    
    k['PtlH0lH1'] = (lepH1+lepH0).Pt()
    k['PtlH1lT'] = (lepH1+lepT).Pt()

    k['dR_lH0_lH1'] = lepH1.DeltaR(lepH0)
    k['dR_lH1_lT'] = lepH1.DeltaR(lepT)
    k['dR_lH0_lT'] = lepH0.DeltaR(lepT)

    k['MllHT0Met'] = (lepH1+lepH0+met).M()
    k['MllHT1Met'] = (lepH1+lepT+met).M()
    k['MllT0T1Met'] = (lepH0+lepT+met).M()

    k['dR_lH0lH1_lT'] = (lepH0+lepH1).DeltaR(lepT)
    k['dR_lH0lT_lH1'] = (lepH0+lepT).DeltaR(lepH1)
    k['dR_lH1lT_lH0'] = (lepH1+lepT).DeltaR(lepH0)

    k['dPhi_lH0lH1_met'] = (lepH0+lepH1).Phi() - met.Phi()
    k['dPhi_lH1lT_met'] = (lepH1+lepT).Phi() - met.Phi()
    k['dPhi_lH0lT_met'] = (lepH0+lepT).Phi() - met.Phi()

    k['met'] = met.Pt()
    k['HT'] = nom.HT

    return k

def higgsDict3lS(nom, jetIdx0, jetIdx1, lepIdx, match=-1):
    '''                                                                                             
    Takes a ttree, two jet indices, and which of the three leptons is considered part of the Higgs candidate (1 or 2).    
    Returns a dict for identifying Higgs decay products                                                     
    Give match if using for training. =1 if these jets, leptons are truth matched to the Higgs, 0 if not        
    '''

    #Initialize four-vectors of decay product candidates                                                                
    jet0, jet1, met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 1, 0)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs     
    if lepIdx == 1:                                                                      
        lepH = lep1
        lepT0 = lep0                                                                                                    
        lepT1 = lep2
    elif lepIdx == 2:                                                                                           
        lepH = lep2                                                                                                 
        lepT0 = lep0
        lepT1 = lep1
    else:                                                                                                         
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['lep_Eta_H'] = lepH.Eta()                                                                                
    k['jet_Eta_0'] = jet0.Eta()                                                                                      
    k['jet_Eta_1'] = jet1.Eta()
    k['lep_Eta_T0'] = lepT0.Eta()
    k['lep_Eta_T1'] = lepT1.Eta()

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    k['Mj0j1'] = (jet0+jet1).M()

    k['dR_lH_j0'] = lepH.DeltaR(jet0)
    k['dR_lH_j1'] = lepH.DeltaR(jet1)

    k['dR_j0j1_lH'] = (jet0 + jet1).DeltaR(lepH)
    k['MjjlH'] = (jet0+jet1+lepH).M()

    k['lep_Pt_T0'] = lepT0.Pt()
    k['dR_j0j1_lT0)'] = (jet0+jet1).DeltaR(lepT0)
    k['Mj0j1lT0'] = (jet0+jet1+lepT0).M()

    k['lep_Pt_T1'] = lepT1.Pt()
    k['dR_j0j1_lT1'] = (jet0+jet1).DeltaR(lepT1)
    k['Mj0j1lT1'] = (jet0+jet1+lepT1).M()

    k['dR_lT0_lT2'] = lepT0.DeltaR(lepT1)
    k['dR_lH_lT0'] = lepH.DeltaR(lepT0)
    k['dR_lH_lT1'] = lepH.DeltaR(lepT1)

    k['jet_DL1r_0'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx0]                                                          
    k['jet_DL1r_1'] = nom.jet_tagWeightBin_DL1r_Continuous[jetIdx1]
    k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85 
    k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60

    k['dR_j0j1lH_lT0'] = (jet0+jet1+lepH).DeltaR(lepT0)
    k['dR_j0j1lH_lT1'] = (jet0+jet1+lepH).DeltaR(lepT1)
    k['dPhi_j0j1lH_met'] = (jet0+jet1+lepH).Phi() - met.Phi()

    k['MET'] = met.Pt()
    k['HT_jets'] = nom.HT_jets
    k['nJets_OR'] = nom.nJets_OR

    return k

def higgsTopDict2lSS(nom, jetIdx0, jetIdx1, lepIdx, topIdx0, topIdx1, topScore, match=-1):
    '''
    Given the indices at two jets and a lepton, plus the indices of top jets, return a dict for predicting whether than 
    pairing of jets+lepton came from a Higgs decay
    '''

    #Get Lorentz vectors of the physics objects
    jet0, jet1, met, lep0, lep1 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 0, 0)
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs
    if lepIdx == 0:                                                                                                          
        lepH = lep0                                                                                                     
        lepT = lep1                                                                                                 
    elif lepIdx == 1:                                                                                                 
        lepH = lep1                                                                                              
        lepT = lep0                                                                                        
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 0 or 1")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['lep_Pt_T'] = lepT.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()

    k['top_Pt_0'] = top0.Pt()   
    k['top_Pt_1'] = top1.Pt()  
    k['top_Eta_0'] = top0.Eta() #might remove
    k['top_Eta_1'] = top1.Eta() # might remove

    k['jet_Eta_0'] = jet0.Eta()
    k['jet_Eta_1'] = jet1.Eta()
    k['jet_Phi_0'] = jet0.Phi()-met.Phi()                                                                            
    k['jet_Phi_1'] = jet1.Phi()-met.Phi()

    k['lep_Eta_H'] = lepH.Eta()                                                                                           
    k['lep_Eta_T'] = lepT.Eta()

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    k['dR_lH_j0'] = lepH.DeltaR(jet0)
    k['dR_lH_j1'] = lepH.DeltaR(jet1)

    k['Mj0j1'] = (jet0+jet1).M()
    k['MlHj0'] = (lepH+jet0).M()
    k['MlHj1'] = (lepH+jet1).M()

    #possibly reinclude
    #k['MlHt0'] = (lepH+top0).M()
    #k['MlHt1'] = (lepH+top1).M()
    #k['Mt0lT'] = (top0+lepT).M() 
    #k['Mt1lT'] = (top1+lepT).M()
    #k['Mj0j1t0'] = (jet0+jet1+top0).M() 
    #k['Mj0j1t1'] = (jet0+jet1+top1).M()

    k['dR_lH_t0'] = lepH.DeltaR(top0)
    k['dR_lH_t1'] = lepH.DeltaR(top1)
    k['dR_lT_t0'] = lepT.DeltaR(top0) 
    k['dR_lT_t1'] = lepT.DeltaR(top1)
    k['dR_j0j1_lH'] = (jet0 + jet1).DeltaR(lepH)
    k['dR_j0j1_lT'] = (jet0 + jet1).DeltaR(lepT)
    k['dR_j0j1_t0'] = (jet0 + jet1).DeltaR(top0)
    k['dR_j0j1_t1'] = (jet0 + jet1).DeltaR(top1)

    #possibly to remove
    k['dR_j0_t0'] = (jet0).DeltaR(top0)
    k['dR_j0_t1'] = (jet0).DeltaR(top1)
    k['dR_j1_t0'] = (jet1).DeltaR(top0)                                                                                     
    k['dR_j1_t1'] = (jet1).DeltaR(top1)

    higgsCand = jet0+jet1+lepH
    k['Mj0j1lH'] = higgsCand.M()
    #k['dR_j0j1lH_t0'] = higgsCand.DeltaR(top0)
    #k['dR_j0j1lH_t1'] = higgsCand.DeltaR(top1)
    #k['dR_j0j1lH_lT'] = higgsCand.DeltaR(lepT)
    #k['dPhi_j0j1lH_met'] = higgsCand.Phi() - met.Phi()

    k['Ptj0j1lHlTt0t1met'] = (top0+top1+jet0+jet1+lepT+lepH+met).Pt() 
    #k['Mj0j1lHlTt0t1met'] = (top0+top1+jet0+jet1+lepT+lepH+met).M() 

    k['topScore'] = topScore
    k['met'] = met.Pt()
    k['nJets_OR'] = nom.nJets_OR
    k['HT_jets'] = nom.HT_jets

    return k

def higgsTopDict3lS(nom, jetIdx0, jetIdx1, lepIdx, topIdx0, topIdx1, topScore, match =-1):
    '''
    '''
    
    jet0, jet1, met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 1, 0)
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs                                 
    if lepIdx == 1:                                                                                                       
        lepH = lep1
        lepT0 = lep0
        lepT1 = lep2
    elif lepIdx == 2:
        lepH = lep2
        lepT0 = lep0
        lepT1 = lep1
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0

    k = {}                                                                                                                  
    if match!=-1:                                                                                                   
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()            
    k['lep_Pt_T0'] = lepT0.Pt()
    k['lep_Pt_T1'] = lepT1.Pt()
    k['jet_Pt_0'] = jet0.Pt()                                                                                              
    k['jet_Pt_1'] = jet1.Pt()                                                                                                
    k['top_Pt_0'] = top0.Pt()                                                                                        
    k['top_Pt_1'] = top1.Pt()

    #k['lep_Eta_H'] = lepH.Eta()                                                                                            
    #k['lep_Eta_T0'] = lepT0.Eta()                                                                                   
    #k['lep_Eta_T1'] = lepT1.Eta()
    k['jet_Eta_0'] = jet0.Eta()                                                                                          
    k['jet_Eta_1'] = jet1.Eta()                                                                                    
    #k['top_Eta_0'] = top0.Eta()                                                                                   
    #k['top_Eta_1'] = top1.Eta()

    k['jet_Phi_0'] = jet0.Phi()-met.Phi()
    k['jet_Phi_1'] = jet1.Phi()-met.Phi()

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    k['Mj0j1'] = (jet0+jet1).M()

    k['dR_lH_j0'] = lepH.DeltaR(jet0)
    k['dR_lH_j1'] = lepH.DeltaR(jet1)
    
    k['dR_j0j1_lH'] = (jet0 + jet1).DeltaR(lepH)
    k['dR_j0j1_lT1'] = (jet0+jet1).DeltaR(lepT1)
    k['dR_lT0_lT1'] = lepT0.DeltaR(lepT1)                                                                             
    k['dR_lH_lT1'] = lepH.DeltaR(lepT1)

    k['Mj0j1lT0'] = (jet0+jet1+lepT0).M()
    k['Mj0j1lT1'] = (jet0+jet1+lepT1).M()

    higgsCand = jet0+jet1+lepH
    k['Mj0j1lH'] = higgsCand.M()
    k['dR_j0j1lH_lT0'] = higgsCand.DeltaR(lepT0)
    k['dR_j0j1lH_lT1'] = higgsCand.DeltaR(lepT1)
    #k['dR_j0j1lH_t0'] = higgsCand.DeltaR(top0)
    #k['dR_j0j1lH_t1'] = higgsCand.DeltaR(top1)
    k['dPhi_j0j1lH_met'] = higgsCand.Phi() - met.Phi()

    k['Ptj0j1lHlT0lT1t0t1met'] = (top0+top1+jet0+jet1+lepT0+lepT1+lepH+met).Pt()
    #k['Mj0j1lHlT0lT1t0t1met'] = (top0+top1+jet0+jet1+lepT0+lepT1+lepH+met).M() 

    #k['Mt0lT0'] = (top0+lepT0).M()
    #k['Mt0lT1'] = (top0+lepT1).M()                                                                                    
    #k['Mt1lT0'] = (top1+lepT0).M()                                                                                   
    #k['Mt1lT1'] = (top1+lepT1).M()
    k['Mj0j1t0'] = (jet0+jet1+top0).M()                                                                     
    k['Mj0j1t1'] = (jet0+jet1+top1).M()
    
    k['dR_lT0_t0'] = lepT0.DeltaR(top0)
    k['dR_lT0_t1'] = lepT0.DeltaR(top1)
    k['dR_lT1_t0'] = lepT1.DeltaR(top0) 
    k['dR_lT1_t1'] = lepT1.DeltaR(top1)
    k['dR_j0_t0'] = (jet0).DeltaR(top0)
    k['dR_j0_t1'] = (jet0).DeltaR(top1)
    k['dR_j1_t0'] = (jet1).DeltaR(top0)
    k['dR_j1_t1'] = (jet1).DeltaR(top1)

    k['topScore'] = topScore
    k['MET'] = met.Pt()
    k['HT_jets'] = nom.HT_jets
    k['nJets_OR'] = nom.nJets_OR

    return k

def higgsTopDict3lF(nom, lepIdx, topIdx0, topIdx1, topScore, match=-1):
    '''                                                                                                                      
    '''

    met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, 0, 0, 1, 1)                                                          
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)                                                             

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs                               
    lepH0 = lep0
    if lepIdx == 1:                                                                                                    
        lepH1 = lep1                                                                                               
        lepT = lep2                                                                                                 
    elif lepIdx == 2:
        lepH1 = lep2
        lepT = lep1
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 1 or 2")
        return 0
        
    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H1'] = lepH1.Pt()                  
    k['lep_Pt_H0'] = lepH0.Pt()
    k['lep_Pt_T'] = lepT.Pt()
    k['top_Pt_0'] = top0.Pt()   
    k['top_Pt_1'] = top1.Pt()  
 
    #k['lep_Eta_H1'] = lepH1.Eta()
    #k['lep_Eta_H0'] = lepH0.Eta()                                                                                           
    #k['lep_Eta_T'] = lepT.Eta()

    #k['lep_Phi_H1'] = lepH1.Phi() - met.Phi()
    #k['lep_Phi_H0'] = lepH0.Phi() - met.Phi()                                                                        
    #k['lep_Phi_T'] = lepT.Phi() - met.Phi()

    k['dPhi_lH1_met'] = lepH1.Phi() - met.Phi()
    k['dPhi_lT_met'] = lepT.Phi() - met.Phi()
   
    k['MlH0lH1'] = (lepH1+lepH0).M()
    k['MlH1lT'] = (lepH1+lepT).M()                                                                                       
    k['MlH0lT'] = (lepH0+lepT).M()

    k['dR_lH0_lH1'] = lepH1.DeltaR(lepH0)
    k['dR_lH1_lT'] = lepH1.DeltaR(lepT)
    k['dR_lH0_lT'] = lepH0.DeltaR(lepT)

    k['MllHT0Met'] = (lepH1+lepH0+met).M()
    k['MllHT1Met'] = (lepH1+lepT+met).M()
    k['MllT0T1Met'] = (lepH0+lepT+met).M()

    k['dR_lH0lH1_lT'] = (lepH0+lepH1).DeltaR(lepT)
    k['dR_lH0lT_lH1'] = (lepH0+lepT).DeltaR(lepH1)

    k['dR_lH0lH1_t0'] = (lepH0+lepH1).DeltaR(top0)
    k['dR_lH0lH1_t1'] = (lepH0+lepH1).DeltaR(top1)

    k['dRlH0t0'] = lepH0.DeltaR(top0) 
    k['MlH0t0'] = (lepH0+top0).M()
    k['dRlH0t1'] = lepH0.DeltaR(top1)  
    k['MlH0t1'] = (lepH0+top1).M()

    k['dRlH1t0'] = lepH1.DeltaR(top0)
    k['MlH1t0'] = (lepH1+top0).M()                                                                                     
    k['dRlH1Ht1'] = lepH1.DeltaR(top1)                                                                                  
    k['MlH1t1'] = (lepH1+top1).M()

    #k['PtlH0lH1lTt0t1met'] = (top0+top1+lepH0+lepH1+lepT+met).Pt()
    #k['MlH0lH1lTt0t1met'] = (top0+top1+lepH0+lepH1+lepT+met).M()

    k['met'] = met.Pt()
    k['topScore'] = topScore     
    #k['HT'] = nom.HT

    return k

def higgsTopDict3lFtest(nom, topIdx0, topIdx1, topScore, match=-1):                                                         
    '''
    '''

    met, lep0, lep1, lep2 = lorentzVecsHiggs(nom, 0, 0, 1, 1)                                                                
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)
    
    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_1'] = lep1.Pt()                                                                               
    k['lep_Pt_0'] = lep0.Pt()
    k['lep_Pt_2'] = lep2.Pt()
    k['top_Pt_0'] = top0.Pt()                                                                                                
    k['top_Pt_1'] = top1.Pt()

    k['Ml0l1'] = (lep1+lep0).M()                                                                                       
    k['Ml1l2'] = (lep1+lep2).M()
    k['Ml0l2'] = (lep0+lep2).M()
    k['dR_l0_l1'] = lep1.DeltaR(lep0)
    k['dR_l1_l2'] = lep1.DeltaR(lep2)
    k['dR_l0_l2'] = lep0.DeltaR(lep2)

    k['Mll20Met'] = (lep1+lep0+met).M()
    k['Mll21Met'] = (lep1+lep2+met).M()
    k['Mll2021Met'] = (lep0+lep2+met).M()

    k['dR_l0l1_l2'] = (lep0+lep1).DeltaR(lep2)
    k['dR_l0l2_l1'] = (lep0+lep2).DeltaR(lep1)

    k['dR_l0l1_t0'] = (lep0+lep1).DeltaR(top0)
    k['dR_l0l1_t1'] = (lep0+lep1).DeltaR(top1)

    k['dRl0t0'] = lep0.DeltaR(top0)
    k['Ml0t0'] = (lep0+top0).M()
    k['dRl0t1'] = lep0.DeltaR(top1)
    k['Ml0t1'] = (lep0+top1).M()

    k['dRl1t0'] = lep1.DeltaR(top0)
    k['Ml1t0'] = (lep1+top0).M()
    k['dRl1t1'] = lep1.DeltaR(top1)
    k['Ml1t1'] = (lep1+top1).M()
    k['Ptl0l1l2t0t1met'] = (top0+top1+lep0+lep1+lep2+met).Pt()
    k['Ml0l1l2t0t1met'] = (top0+top1+lep0+lep1+lep2+met).M()

    k['met'] = met.Pt()                                                                                                      
    k['topScore'] = topScore
    k['HT'] = nom.HT
    return k

def higgsTopDictBig2lSS(nom, jetIdx0, jetIdx1, lepIdx, topIdx0, topIdx1, topScore, match=-1):
    '''
    Given the indices at two jets and a lepton, plus the indices of top jets, return a dict for predicting whether than
    pairing of jets+lepton came from a Higgs decay
    '''

    #Get Lorentz vectors of the physics objects
    jet0, jet1, met, lep0, lep1 = lorentzVecsHiggs(nom, jetIdx0, jetIdx1, 0, 0)
    top0, top1 = lorentzVecsTop(nom, topIdx0, topIdx1)

    #Leptons come from top or higgs. Identify which we want to consider to be from the Higgs
    if lepIdx == 0:
        lepH = lep0
        lepT = lep1
    elif lepIdx == 1:
        lepH = lep1
        lepT = lep0
    else:
        print(f"{lepIdx} is not a valid lep index. Must be 0 or 1")
        return 0

    k = {}
    if match!=-1:
        k['match'] = match

    k['lep_Pt_H'] = lepH.Pt()
    k['lep_Pt_T'] = lepT.Pt()
    k['jet_Pt_0'] = jet0.Pt()
    k['jet_Pt_1'] = jet1.Pt()
    k['top_Pt_0'] = top0.Pt()
    k['top_Pt_1'] = top1.Pt()

    k['lep_Eta_H'] = lepH.Eta() 
    k['lep_Eta_T'] = lepT.Eta()
    k['jet_Eta_0'] = jet0.Eta()                                                                                           
    k['jet_Eta_1'] = jet1.Eta()
    k['top_Eta_0'] = top0.Eta()                                                                                         
    k['top_Eta_1'] = top1.Eta()

    k['lep_E_H'] = lepH.E()
    k['lep_E_T'] = lepT.E()
    k['jet_E_0'] = jet0.E()
    k['jet_E_1'] = jet1.E()
    k['top_E_0'] = top0.E()                                                                                              
    k['top_E_1'] = top1.E()

    k['lep_Phi_H'] = lepH.Phi() - met.Phi()
    k['lep_Phi_T'] = lepT.Phi() - met.Phi()
    k['jet_Phi_0'] = jet0.Phi() - met.Phi()
    k['jet_Phi_1'] = jet1.Phi() - met.Phi()
    k['top_Phi_0'] = top0.Phi() - met.Phi()
    k['top_Phi_1'] = top1.Phi() - met.Phi()

    k['dR_j0_j1'] = jet0.DeltaR(jet1)
    k['dR_lH_j0'] = lepH.DeltaR(jet0)
    k['dR_lH_j1'] = lepH.DeltaR(jet1)

    k['Mj0j1'] = (jet0+jet1).M()
    k['MlHj0'] = (lepH+jet0).M()
    k['MlHj1'] = (lepH+jet1).M()

    k['MlHt0'] = (lepH+top0).M()
    k['MlHt1'] = (lepH+top1).M()
    k['Mt0lT'] = (top0+lepT).M()
    k['Mt1lT'] = (top1+lepT).M()
    k['Mj0j1t0'] = (jet0+jet1+top0).M()
    k['Mj0j1t1'] = (jet0+jet1+top1).M()

    k['dR_lH_t0'] = lepH.DeltaR(top0)
    k['dR_lH_t1'] = lepH.DeltaR(top1)
    k['dR_lT_t0'] = lepT.DeltaR(top0)
    k['dR_lT_t1'] = lepT.DeltaR(top1)
    k['dR_j0j1_lH'] = (jet0 + jet1).DeltaR(lepH)
    k['dR_j0j1_lT'] = (jet0 + jet1).DeltaR(lepT)

    higgsCand = jet0+jet1+lepH
    k['Mj0j1lH'] = higgsCand.M()
    k['dR_j0j1lH_t0'] = higgsCand.DeltaR(top0)
    k['dR_j0j1lH_t1'] = higgsCand.DeltaR(top1)
    k['dR_j0j1lH_lT'] = higgsCand.DeltaR(lepT)
    k['dPhi_j0j1lH_met'] = higgsCand.Phi() - met.Phi()

    k['Ptj0j1lHlTt0t1met'] = (top0+top1+jet0+jet1+lepT+lepH+met).Pt()
    k['Ptj0j1lHlTt0t1met'] = (top0+top1+jet0+jet1+lepT+lepH+met).M()

    k['topScore'] = topScore
    k['met'] = met.Pt()
    k['nJets_OR'] = nom.nJets_OR
    k['HT_jets'] = nom.HT_jets

    return k
